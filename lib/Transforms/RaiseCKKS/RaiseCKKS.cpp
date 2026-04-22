#include "lib/Transforms/RaiseCKKS/RaiseCKKS.h"

#include <cstdint>

#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Orion/IR/OrionOps.h"
#include "lib/Dialect/Orion/IR/OrionUtils.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialDialect.h"
#include "lib/Dialect/Polynomial/IR/PolynomialOps.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/IRMapping.h"              // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_RAISECKKS
#include "lib/Transforms/RaiseCKKS/RaiseCKKS.h.inc"

namespace {

// Get the number of slots from an LWE ciphertext type.
int64_t getSlotCount(lwe::LWECiphertextType ctType) {
  auto ciphertextSpace = ctType.getCiphertextSpace();
  auto ring = ciphertextSpace.getRing();
  auto polyModulus = ring.getPolynomialModulus().getPolynomial();
  // Ring degree is the degree of the polynomial modulus (x^N + 1 has degree N)
  int64_t ringDegree = polyModulus.getDegree();
  return ringDegree / 2;
}

// Get the raised tensor type for an LWE ciphertext type.
Type getRaisedType(lwe::LWECiphertextType ctType) {
  int64_t slots = getSlotCount(ctType);
  // Use f64 since Orion IR uses f64 coefficients
  return RankedTensorType::get({slots}, Float64Type::get(ctType.getContext()));
}

// Convert an LWE type (possibly tensor-wrapped) to a raised type.
Type raiseType(Type type) {
  if (auto ctType = dyn_cast<lwe::LWECiphertextType>(type)) {
    return getRaisedType(ctType);
  }
  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    if (auto ctType =
            dyn_cast<lwe::LWECiphertextType>(shapedType.getElementType())) {
      // tensor<1x!lwe.ct> -> tensor<Nxf64>
      return getRaisedType(ctType);
    }
  }
  return type;
}

struct RaiseCKKS : impl::RaiseCKKSBase<RaiseCKKS> {
  using RaiseCKKSBase::RaiseCKKSBase;

  void runOnOperation() override {
    auto module = getOperation();

    // Preserve the original chain length as a minimum level hint for
    // generate-param-ckks. The original Orion parameters may include extra
    // levels for noise headroom beyond what the circuit strictly needs.
    if (auto schemeParamAttr = module->getAttrOfType<ckks::SchemeParamAttr>(
            ckks::CKKSDialect::kSchemeParamAttrName)) {
      int64_t originalQSize = schemeParamAttr.getQ().size();
      module->setAttr(
          "heir.min_level_hint",
          IntegerAttr::get(IntegerType::get(module->getContext(), 64),
                           originalQSize - 1));
    }
    // Strip CKKS scheme param (will be re-derived by generate-param)
    module->removeAttr(ckks::CKKSDialect::kSchemeParamAttrName);
    // Strip scale policy and other management attrs
    module->removeAttr("ckks.scale_policy");
    module->removeAttr("ckks.reduced_error");
    module->removeAttr("openfhe.scaling_technique");
    module->removeAttr("backend.openfhe");
    module->removeAttr("backend.lattigo");
    module->removeAttr("backend.cheddar");
    module->removeAttr("scheme.requested_slot_count");
    module->removeAttr("scheme.actual_slot_count");

    // Process each function
    auto result = module->walk([&](func::FuncOp func) -> WalkResult {
      if (failed(raiseFunction(func))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }

  LogicalResult raiseFunction(func::FuncOp func) {
    OpBuilder builder(func);

    // Determine raised argument types: plain tensors with secret attribute,
    // NOT secret.secret<> types. The wrap-generic pass will handle wrapping.
    SmallVector<Type> newArgTypes;
    SmallVector<bool> isSecretArg;
    for (Type argType : func.getArgumentTypes()) {
      Type raised = raiseType(argType);
      if (raised != argType) {
        newArgTypes.push_back(raised);  // plain tensor, not secret-wrapped
        isSecretArg.push_back(true);
      } else {
        newArgTypes.push_back(argType);
        isSecretArg.push_back(false);
      }
    }

    // Determine raised result types: also plain tensors
    SmallVector<Type> newResultTypes;
    for (Type resultType : func.getResultTypes()) {
      Type raised = raiseType(resultType);
      if (raised != resultType) {
        newResultTypes.push_back(raised);  // plain tensor
      } else {
        newResultTypes.push_back(resultType);
      }
    }

    // Create new function with raised signature
    auto newFuncType =
        FunctionType::get(func.getContext(), newArgTypes, newResultTypes);

    // Build the new function body with plain arith/tensor ops.
    // The wrap-generic pass will create secret.generic later.
    builder.setInsertionPoint(func);
    auto newFunc = func::FuncOp::create(builder, func.getLoc(), func.getName(),
                                        newFuncType);
    for (auto attr : func->getDialectAttrs()) {
      newFunc->setAttr(attr.getName(), attr.getValue());
    }
    for (unsigned i = 0; i < newArgTypes.size(); ++i) {
      if (isSecretArg[i]) {
        newFunc.setArgAttr(i, "secret.secret", builder.getUnitAttr());
      }
    }

    Block* entryBlock = newFunc.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Map old function args to new function args
    IRMapping mapping;
    for (unsigned i = 0; i < func.getNumArguments(); ++i) {
      mapping.map(func.getArgument(i), entryBlock->getArgument(i));
    }

    // Walk the original function body and raise each op
    for (Operation& op : func.getBody().front()) {
      if (auto returnOp = dyn_cast<func::ReturnOp>(&op)) {
        SmallVector<Value> returnValues;
        for (Value operand : returnOp.getOperands()) {
          returnValues.push_back(mapping.lookup(operand));
        }
        func::ReturnOp::create(builder, op.getLoc(), returnValues);
        continue;
      }

      if (failed(raiseOp(builder, &op, mapping))) {
        return op.emitOpError() << "failed to raise op";
      }
    }

    func.erase();
    return success();
  }

  LogicalResult raiseOp(OpBuilder& builder, Operation* op, IRMapping& mapping) {
    // Strip management ops (rescale, relinearize) - they become identity
    if (isa<ckks::RescaleOp, ckks::RelinearizeOp>(op)) {
      // Map result to input (identity)
      mapping.map(op->getResult(0), mapping.lookup(op->getOperand(0)));
      return success();
    }

    // Binary ciphertext-ciphertext ops
    if (isa<ckks::AddOp>(op)) {
      Value lhs = mapping.lookup(op->getOperand(0));
      Value rhs = mapping.lookup(op->getOperand(1));
      auto result = arith::AddFOp::create(builder, op->getLoc(), lhs, rhs);
      mapping.map(op->getResult(0), result);
      return success();
    }
    if (isa<ckks::SubOp>(op)) {
      Value lhs = mapping.lookup(op->getOperand(0));
      Value rhs = mapping.lookup(op->getOperand(1));
      auto result = arith::SubFOp::create(builder, op->getLoc(), lhs, rhs);
      mapping.map(op->getResult(0), result);
      return success();
    }
    if (isa<ckks::MulOp>(op)) {
      Value lhs = mapping.lookup(op->getOperand(0));
      Value rhs = mapping.lookup(op->getOperand(1));
      auto result = arith::MulFOp::create(builder, op->getLoc(), lhs, rhs);
      mapping.map(op->getResult(0), result);
      return success();
    }
    if (isa<ckks::NegateOp>(op)) {
      Value input = mapping.lookup(op->getOperand(0));
      auto result = arith::NegFOp::create(builder, op->getLoc(), input);
      mapping.map(op->getResult(0), result);
      return success();
    }

    // Rotation
    if (auto rotateOp = dyn_cast<ckks::RotateOp>(op)) {
      Value input = mapping.lookup(rotateOp.getInput());
      Value shift = arith::ConstantIndexOp::create(
          builder, op->getLoc(),
          rotateOp.getStaticShift()->getValue().getSExtValue());
      auto result =
          tensor_ext::RotateOp::create(builder, op->getLoc(), input, shift);
      mapping.map(op->getResult(0), result);
      return success();
    }

    // Plaintext ops: fold encode + add_plain into arith.addf
    if (isa<ckks::AddPlainOp>(op)) {
      Value ct = mapping.lookup(op->getOperand(0));
      Value pt = mapping.lookup(op->getOperand(1));
      auto result = arith::AddFOp::create(builder, op->getLoc(), ct, pt);
      mapping.map(op->getResult(0), result);
      return success();
    }
    if (isa<ckks::SubPlainOp>(op)) {
      Value ct = mapping.lookup(op->getOperand(0));
      Value pt = mapping.lookup(op->getOperand(1));
      auto result = arith::SubFOp::create(builder, op->getLoc(), ct, pt);
      mapping.map(op->getResult(0), result);
      return success();
    }
    if (isa<ckks::MulPlainOp>(op)) {
      Value ct = mapping.lookup(op->getOperand(0));
      Value pt = mapping.lookup(op->getOperand(1));
      auto result = arith::MulFOp::create(builder, op->getLoc(), ct, pt);
      mapping.map(op->getResult(0), result);
      return success();
    }

    // Encode: map to the cleartext input directly
    if (auto encodeOp = dyn_cast<lwe::RLWEEncodeOp>(op)) {
      Value input = mapping.lookup(encodeOp.getInput());
      mapping.map(encodeOp.getResult(), input);
      return success();
    }

    // Orion chebyshev -> polynomial.eval
    if (auto chebyshevOp = dyn_cast<orion::ChebyshevOp>(op)) {
      Value input = mapping.lookup(chebyshevOp.getInput());

      // Build Chebyshev polynomial attribute
      auto polyType = polynomial::PolynomialType::get(
          builder.getContext(),
          polynomial::RingAttr::get(builder.getF64Type()));
      auto typedChebAttr = polynomial::TypedChebyshevPolynomialAttr::get(
          polyType, chebyshevOp.getCoefficientsAttr());

      auto evalOp = polynomial::EvalOp::create(builder, op->getLoc(),
                                               typedChebAttr, input);

      // Preserve domain info
      evalOp->setAttr("domain_lower", chebyshevOp.getDomainStartAttr());
      evalOp->setAttr("domain_upper", chebyshevOp.getDomainEndAttr());

      mapping.map(op->getResult(0), evalOp.getResult());
      return success();
    }

    // Orion linear_transform -> linalg.matvec
    if (auto linearTransformOp = dyn_cast<orion::LinearTransformOp>(op)) {
      Value input = mapping.lookup(linearTransformOp.getInput());
      int64_t slots = linearTransformOp.getSlots().getValue().getSExtValue();
      auto diagIndices = linearTransformOp.getDiagonalIndices();
      int64_t diagCount = diagIndices.size();

      // Try to get compile-time constant diagonals for matrix reconstruction.
      Value diagonals = linearTransformOp.getDiagonals();
      DenseElementsAttr diagData;
      if (auto constOp = diagonals.getDefiningOp<arith::ConstantOp>()) {
        diagData = dyn_cast<DenseElementsAttr>(constOp.getValue());
      }
      if (!diagData) {
        // Function-parameter diagonals: emit tensor_ext.diagonal_matvec
        // which preserves the diagonal representation through management.
        Value mappedDiags = mapping.lookup(diagonals);
        auto result = tensor_ext::DiagonalMatvecOp::create(
            builder, op->getLoc(), input.getType(), input, mappedDiags,
            linearTransformOp.getDiagonalIndicesAttr(),
            builder.getI64IntegerAttr(slots));
        mapping.map(op->getResult(0), result);
        return success();
      }

      auto elementType =
          cast<RankedTensorType>(diagonals.getType()).getElementType();

      // Reconstruct weight matrix from diagonals.
      // Convention: diagonal_k[j] = M[j][(k + j) % slots]
      SmallVector<double> matrixValues(slots * slots, 0.0);
      for (int64_t i = 0; i < diagCount; ++i) {
        int64_t diagIdx = diagIndices[i];
        for (int64_t j = 0; j < slots; ++j) {
          int64_t col = ((diagIdx + j) % slots + slots) % slots;
          matrixValues[j * slots + col] = diagData.getValues<double>()[{
              static_cast<unsigned>(i), static_cast<unsigned>(j)}];
        }
      }

      auto matrixType = RankedTensorType::get({slots, slots}, elementType);
      Value matrix = arith::ConstantOp::create(
          builder, op->getLoc(), matrixType,
          DenseFPElementsAttr::get(matrixType, matrixValues));

      auto vectorType = cast<RankedTensorType>(input.getType());
      Value initZero = arith::ConstantOp::create(
          builder, op->getLoc(), vectorType,
          DenseFPElementsAttr::get(vectorType,
                                   SmallVector<double>(slots, 0.0)));
      auto matvecResult = linalg::MatvecOp::create(builder, op->getLoc(),
                                                   ValueRange{matrix, input},
                                                   ValueRange{initZero});

      mapping.map(op->getResult(0), matvecResult.getResult(0));
      return success();
    }

    // Cleartext ops (arith.constant, tensor ops, etc.) that don't involve
    // ciphertext types: clone them as-is into the raised IR.
    bool hasCiphertextOperand = false;
    for (Type type : op->getOperandTypes()) {
      if (isa<lwe::LWECiphertextType>(getElementTypeOrSelf(type))) {
        hasCiphertextOperand = true;
        break;
      }
    }
    if (!hasCiphertextOperand) {
      auto* cloned = builder.clone(*op, mapping);
      for (unsigned i = 0; i < op->getNumResults(); ++i) {
        mapping.map(op->getResult(i), cloned->getResult(i));
      }
      return success();
    }

    return op->emitOpError() << "unsupported op in raise-ckks";
  }
};

}  // namespace

}  // namespace heir
}  // namespace mlir
