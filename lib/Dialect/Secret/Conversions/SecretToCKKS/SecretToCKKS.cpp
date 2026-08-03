#include "lib/Dialect/Secret/Conversions/SecretToCKKS/SecretToCKKS.h"

#include <cassert>
#include <cstdint>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSEnums.h"
#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/Orion/IR/OrionDialect.h"
#include "lib/Dialect/Orion/IR/OrionOps.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialOps.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"
#include "lib/Dialect/Secret/Conversions/Patterns.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Utils/AttributeUtils.h"
#include "lib/Utils/ContextAwareConversionUtils.h"
#include "lib/Utils/ContextAwareDialectConversion.h"
#include "lib/Utils/ContextAwareTypeConversion.h"
#include "lib/Utils/Polynomial/Polynomial.h"
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"       // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AsmState.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"            // from @llvm-project
#include "mlir/include/mlir/IR/DialectResourceBlobManager.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Matchers.h"            // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"        // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"               // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

// IWYU pragma: begin_keep
#include "lib/Dialect/LWE/IR/LWEDialect.h"
// IWYU pragma: end_keep

namespace mlir::heir {

#define GEN_PASS_DEF_SECRETTOCKKS
#include "lib/Dialect/Secret/Conversions/SecretToCKKS/SecretToCKKS.h.inc"

namespace {

// Returns an RLWE ring given the specified number of bits needed and polynomial
// modulus degree.
FailureOr<polynomial::RingAttr> getRlweRNSRing(
    MLIRContext* ctx, const std::vector<int64_t>& primes, int minSlotCount) {
  // monomial
  std::vector<polynomial::IntMonomial> monomials;
  monomials.emplace_back(1, minSlotCount);
  monomials.emplace_back(1, 0);
  auto result = polynomial::IntPolynomial::fromMonomials(monomials);
  if (failed(result)) return failure();
  polynomial::IntPolynomial xnPlusOne = result.value();

  // moduli chain
  SmallVector<Type, 4> modTypes;
  for (auto prime : primes) {
    auto type = IntegerType::get(ctx, 64);
    modTypes.push_back(
        mod_arith::ModArithType::get(ctx, IntegerAttr::get(type, prime)));
  }

  // types
  auto rnsType = rns::RNSType::get(ctx, modTypes);
  return polynomial::RingAttr::get(
      rnsType, polynomial::IntPolynomialAttr::get(ctx, xnPlusOne));
}

polynomial::RingAttr getRlweRNSRingWithLevel(polynomial::RingAttr ringAttr,
                                             int level) {
  auto rnsType = cast<rns::RNSType>(ringAttr.getCoefficientType());

  auto newRnsType = rns::RNSType::get(
      rnsType.getContext(), rnsType.getBasisTypes().take_front(level + 1));
  return polynomial::RingAttr::get(newRnsType, ringAttr.getPolynomialModulus());
}

}  // namespace

class SecretToCKKSTypeConverter
    : public UniquelyNamedAttributeAwareTypeConverter {
 public:
  SecretToCKKSTypeConverter(MLIRContext* ctx, polynomial::RingAttr rlweRing)
      : UniquelyNamedAttributeAwareTypeConverter(
            mgmt::MgmtDialect::kArgMgmtAttrName) {
    addConversion([](Type type, Attribute attr) { return type; });
    addConversion(
        [this](RankedTensorType type, mgmt::MgmtAttr mgmtAttr) -> Type {
          // For cases like tensor.empty + mgmt.init, we need to convert this
          // to a ciphertext type.
          //
          // Care must be taken to ensure that types that have already been
          // converted (i.e., their element type is a ciphertext or plaintext
          // type) are returned as-is, or else legality checking will consider
          // ops with these types illegal.
          if (isa<lwe::LWECiphertextType, lwe::LWEPlaintextType>(
                  type.getElementType()))
            return type;
          return convertSecretTypeWithMgmtAttr(secret::SecretType::get(type),
                                               mgmtAttr);
        });
    addConversion([this](secret::SecretType type, mgmt::MgmtAttr mgmtAttr) {
      return convertSecretTypeWithMgmtAttr(type, mgmtAttr);
    });

    ring_ = rlweRing;
  }

  Type convertSecretTypeWithMgmtAttr(secret::SecretType type,
                                     mgmt::MgmtAttr mgmtAttr) const {
    auto* ctx = type.getContext();
    auto level = mgmtAttr.getLevel();
    auto dimension = mgmtAttr.getDimension();
    auto scale = mgmtAttr.getScale();
    auto tensorValueType = dyn_cast<RankedTensorType>(type.getValueType());

    // Note that slot number for CKKS is always half of the ring dimension.
    // so ring_.getPolynomialModulus() is not useful here
    // TODO(#2764): use packing information to get the correct slot number
    auto plaintextRing = polynomial::RingAttr::get(
        ctx, Float64Type::get(ctx), ring_.getPolynomialModulus());

    SmallVector<IntegerAttr, 6> modulusChain;
    for (auto modArithType :
         cast<rns::RNSType>(ring_.getCoefficientType()).getBasisTypes()) {
      auto modulus = cast<mod_arith::ModArithType>(modArithType).getModulus();
      modulusChain.push_back(modulus);
    }

    Type messageType = type.getValueType();
    if (tensorValueType && tensorValueType.getRank() > 1) {
      // The value type here is a ciphertext-semantic tensor (i.e., packed) and
      // so the "message type" is what is packed in each ciphertext, i.e., a
      // single dimensional tensor corresponding to the last axis.
      // TODO(#2280): this is where we are forced to use the packed cleartexts
      messageType = RankedTensorType::get(
          {tensorValueType.getDimSize(tensorValueType.getRank() - 1)},
          tensorValueType.getElementType());
    }

    auto ctType = lwe::LWECiphertextType::get(
        ctx,
        lwe::PlaintextSpaceAttr::get(
            ctx, plaintextRing,
            lwe::InverseCanonicalEncodingAttr::get(ctx, scale)),
        lwe::CiphertextSpaceAttr::get(ctx,
                                      getRlweRNSRingWithLevel(ring_, level),
                                      lwe::LweEncryptionType::mix, dimension),
        lwe::KeyAttr::get(ctx, 0),
        lwe::ModulusChainAttr::get(ctx, modulusChain, level));

    if (tensorValueType) {
      // A rank-1 tensor is interpreted as a single ciphertext
      if (tensorValueType.getRank() == 1) {
        return ctType;
      }

      // A rank-2+ tensor is a tensor of ciphertexts, where the last axis
      // becomes the ciphertext type. This is the most common case where
      // data is packed in a set of ciphertexts.
      return RankedTensorType::get(tensorValueType.getShape().drop_back(),
                                   ctType);
    }

    return ctType;
  }

 private:
  polynomial::RingAttr ring_;
};

class SecretGenericPlaintextDivision
    : public SecretGenericOpConversion<arith::DivFOp, ckks::MulPlainOp> {
 public:
  using SecretGenericOpConversion<arith::DivFOp,
                                  ckks::MulPlainOp>::SecretGenericOpConversion;

  FailureOr<Operation*> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ContextAwareConversionPatternRewriter& rewriter) const override {
    // Check that the divisor is a plaintext.
    Value ciphertextInput = inputs[0];
    Value cleartextDivisor = inputs[1];
    if (isa<lwe::LWECiphertextType>(cleartextDivisor.getType()) ||
        !isa<lwe::LWECiphertextType>(ciphertextInput.getType())) {
      return rewriter.notifyMatchFailure(
          op, "expected plaintext divisor and ciphertext dividend");
    }

    // Encode 1/cleartext as a plaintext
    auto initOp =
        dyn_cast_or_null<mgmt::InitOp>(cleartextDivisor.getDefiningOp());
    if (!initOp) {
      return rewriter.notifyMatchFailure(
          op, "expected plaintext divisor to be defined by mgmt.init");
    }
    auto mgmtAttr = mgmt::findMgmtAttrAssociatedWith(initOp);
    if (!mgmtAttr) {
      return rewriter.notifyMatchFailure(
          op, "expected plaintext divisor to have mgmt.mgmt attribute");
    }

    Value realCleartext = initOp.getInput();
    Value invertedCleartext = arith::DivFOp::create(
        rewriter, op.getLoc(),
        arith::ConstantOp::create(rewriter, op.getLoc(),
                                  realCleartext.getType(),
                                  rewriter.getOneAttr(realCleartext.getType())),
        realCleartext);

    std::string errMsg;
    llvm::raw_string_ostream errStream(errMsg);
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto ciphertextElementTy = cast<lwe::LWECiphertextType>(
        getElementTypeOrSelf(ciphertextInput.getType()));
    FailureOr<Value> encodedPlaintext = encodeCleartextAsPlaintext(
        b, invertedCleartext, ciphertextElementTy, mgmtAttr, errStream);
    if (failed(encodedPlaintext)) {
      return rewriter.notifyMatchFailure(op, errMsg);
    }

    return rewriter
        .replaceOpWithNewOp<ckks::MulPlainOp>(op, ciphertextInput,
                                              encodedPlaintext.value())
        .getOperation();
  }
};

// Lower a kept `polynomial.eval` (Chebyshev basis) to `orion.chebyshev`, which
// the Lattigo backend emits as a single compact polynomial.Evaluate call
// instead of an unrolled Paterson-Stockmeyer mul/add chain. Only fires under
// --preserve-poly-eval, which skips LowerPolynomialEval so the eval op survives
// to here. The op's multiplicative depth was already accounted for by mgmt via
// ReducesLevelOpInterface (see Polynomial EvalOp::getLevelsToDrop).
class ConvertChebyshevEval
    : public SecretGenericOpConversion<polynomial::EvalOp, orion::ChebyshevOp> {
 public:
  using SecretGenericOpConversion<
      polynomial::EvalOp, orion::ChebyshevOp>::SecretGenericOpConversion;

  FailureOr<Operation*> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ContextAwareConversionPatternRewriter& rewriter) const override {
    auto evalOp =
        cast<polynomial::EvalOp>(op.getBody()->getOperations().front());
    auto chebAttr = dyn_cast<polynomial::TypedChebyshevPolynomialAttr>(
        evalOp.getPolynomialAttr());
    if (!chebAttr) {
      return rewriter.notifyMatchFailure(
          evalOp, "orion.chebyshev lowering requires a Chebyshev-basis poly");
    }
    auto lower = evalOp->getAttrOfType<FloatAttr>("domain_lower");
    auto upper = evalOp->getAttrOfType<FloatAttr>("domain_upper");
    if (!lower || !upper) {
      return rewriter.notifyMatchFailure(
          evalOp, "polynomial.eval missing domain_lower/domain_upper attrs");
    }
    auto chebyOp = orion::ChebyshevOp::create(
        rewriter, evalOp.getLoc(), outputTypes[0], inputs[0],
        chebAttr.getValue().getCoefficients(), lower, upper,
        /*output_scale=*/IntegerAttr());
    rewriter.replaceOp(op, chebyOp);
    return chebyOp.getOperation();
  }
};

// Returns the raw little-endian element data of a constant's value attribute,
// or failure when it isn't accessible (splat-compressed attrs, unresolved
// dense resources, non-dense attrs).
static FailureOr<ArrayRef<char>> rawConstantData(Attribute value) {
  if (auto dense = dyn_cast<DenseIntOrFPElementsAttr>(value)) {
    if (dense.isSplat()) return failure();
    return dense.getRawData();
  }
  if (auto resource = dyn_cast<DenseResourceElementsAttr>(value)) {
    if (AsmResourceBlob* blob = resource.getRawHandle().getBlob())
      return blob->getData();
  }
  return failure();
}

// Builds a dense attr holding the given rows of a [*, cols] constant whose
// raw data is `raw`.
static DenseElementsAttr gatherRows(ArrayRef<char> raw,
                                    ArrayRef<int32_t> rowIndices, int64_t cols,
                                    Type elementType) {
  int64_t elemBytes = elementType.getIntOrFloatBitWidth() / 8;
  int64_t rowBytes = cols * elemBytes;
  SmallVector<char> gathered;
  gathered.reserve(rowIndices.size() * rowBytes);
  for (int32_t row : rowIndices) {
    ArrayRef<char> rowData = raw.slice((int64_t)row * rowBytes, rowBytes);
    gathered.append(rowData.begin(), rowData.end());
  }
  auto type = RankedTensorType::get(
      {static_cast<int64_t>(rowIndices.size()), cols}, elementType);
  return DenseElementsAttr::getFromRawBuffer(type, gathered);
}

// A lintrans-marked tensor_ext.rotate_and_reduce (produced by
// convert-to-ciphertext-semantics under use-lintrans-kernels) lowers to
// orion.linear_transform, evaluated by the backend's optimized
// linear-transform kernel (cheddar/cyclops LinearTransform, lattigo
// lintrans). Mirrors ConvertChebyshevEval.
class ConvertRotateAndReduce
    : public SecretGenericOpConversion<tensor_ext::RotateAndReduceOp,
                                       orion::LinearTransformOp> {
 public:
  using SecretGenericOpConversion<
      tensor_ext::RotateAndReduceOp,
      orion::LinearTransformOp>::SecretGenericOpConversion;

  FailureOr<Operation*> matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ContextAwareConversionPatternRewriter& rewriter) const override {
    auto rarOp = cast<tensor_ext::RotateAndReduceOp>(
        op.getBody()->getOperations().front());
    if (!rarOp->hasAttr(tensor_ext::TensorExtDialect::kLintransAttrName)) {
      return rewriter.notifyMatchFailure(
          op,
          "only lintrans-marked rotate_and_reduce lowers to "
          "orion.linear_transform");
    }
    if (!rarOp.getPlaintexts()) {
      return rewriter.notifyMatchFailure(
          op,
          "rotate_and_reduce without plaintext diagonals cannot lower to "
          "orion.linear_transform");
    }

    // `inputs` follows the inner op's operand order: [ciphertext, diagonals].
    // The diagonals may be wrapped in an mgmt.init placeholder; the backend
    // kernel encodes the raw float tensor itself, so unwrap it.
    Value input = inputs[0];
    Value diagonals = inputs[1];
    if (auto initOp = diagonals.getDefiningOp<mgmt::InitOp>()) {
      diagonals = initOp.getInput();
    }

    auto diagType = cast<RankedTensorType>(diagonals.getType());
    int64_t numDiagonals = diagType.getDimSize(0);
    int64_t slots = diagType.getDimSize(1);

    // The transform evaluates at the input ciphertext's level.
    auto ctTy =
        dyn_cast<lwe::LWECiphertextType>(getElementTypeOrSelf(input.getType()));
    if (!ctTy || !ctTy.getModulusChain()) {
      return rewriter.notifyMatchFailure(
          op, "input ciphertext has no modulus chain (run level analysis)");
    }
    int64_t level = ctTy.getModulusChain().getCurrent();

    // A conv kernel expanded to a diagonal matvec (Toeplitz matrix) is mostly
    // all-zero diagonals: convert-to-ciphertext-semantics records the nonzero
    // rows' offsets in a tensor_ext.diagonal_indices attr. When the diagonal
    // tensor is a compile-time constant (the usual case: assign_layout
    // materializes weights at compile time), compact it down to those rows,
    // so the backend kernel neither encodes nor rotates for zeros. Row i of
    // the compacted tensor is diagonal diagonalIndices[i] — the positional
    // convention the orion frontend and the backend emitters use, which is
    // also why the full tensor must keep the full index range: without
    // compaction the rows would be misattributed.
    SmallVector<int32_t> diagonalIndices;
    if (auto indicesAttr = rarOp->getAttrOfType<DenseI32ArrayAttr>(
            tensor_ext::TensorExtDialect::kDiagonalIndicesAttrName)) {
      if (auto cstOp = diagonals.getDefiningOp<arith::ConstantOp>()) {
        FailureOr<ArrayRef<char>> raw = rawConstantData(cstOp.getValue());
        if (succeeded(raw)) {
          ArrayRef<int32_t> indices = indicesAttr.asArrayRef();
          DenseElementsAttr compacted =
              gatherRows(*raw, indices, slots, diagType.getElementType());
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointAfter(cstOp);
          diagonals =
              arith::ConstantOp::create(rewriter, cstOp.getLoc(), compacted)
                  .getResult();
          diagonalIndices.assign(indices.begin(), indices.end());
        }
      }
    }
    if (diagonalIndices.empty()) {
      // Dense weights, or diagonals we cannot compact (runtime-materialized):
      // every row of the tensor is its own diagonal.
      diagonalIndices.resize(numDiagonals);
      std::iota(diagonalIndices.begin(), diagonalIndices.end(), 0);
    }

    auto lintransOp = orion::LinearTransformOp::create(
        rewriter, op.getLoc(), outputTypes[0], input, diagonals,
        rewriter.getDenseI32ArrayAttr(diagonalIndices),
        /*orion_level=*/rewriter.getI32IntegerAttr(level),
        /*bsgs_ratio=*/rewriter.getF64FloatAttr(2.0),
        /*slots=*/rewriter.getI32IntegerAttr(slots));
    rewriter.replaceOp(op, lintransOp);
    return lintransOp.getOperation();
  }
};

struct SecretToCKKS : public impl::SecretToCKKSBase<SecretToCKKS> {
  using SecretToCKKSBase::SecretToCKKSBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    auto schemeParamAttr = module->getAttrOfType<ckks::SchemeParamAttr>(
        ckks::CKKSDialect::kSchemeParamAttrName);
    if (!schemeParamAttr) {
      module->emitError("expected CKKS scheme parameters");
      signalPassFailure();
      return;
    }

    // NOTE: 2 ** logN != minSlotCount
    // they have different semantic
    // auto logN = schemeParamAttr.getLogN();

    // pass option minSlotCount is actually the number of slots
    // TODO(#1402): use a proper name for CKKS
    auto rlweRing = getRlweRNSRing(context, schemeParamAttr.getQ().asArrayRef(),
                                   minSlotCount);
    if (failed(rlweRing)) {
      return signalPassFailure();
    }

    bool usePublicKey =
        schemeParamAttr.getEncryptionType() == ckks::CKKSEncryptionType::pk;

    // Invariant: for every SecretType, there is a
    // corresponding MgmtAttr attached to it,
    // either in its DefiningOp or getOwner()->getParentOp()
    // (i.e., the FuncOp).
    // Otherwise the typeConverter won't find the proper type information
    // and fail
    SecretToCKKSTypeConverter typeConverter(context, rlweRing.value());
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    addSecretToSchemeDefaultConversionTargetsAndPatterns(patterns, target,
                                                         typeConverter);

    target.addLegalDialect<ckks::CKKSDialect>();
    // orion.chebyshev is produced under --preserve-poly-eval and lowered to a
    // compact backend call downstream. (The Orion dialect is also marked legal
    // so any explicit orion.linear_transform in the input IR survives.)
    target.addLegalDialect<orion::OrionDialect>();
    patterns.add<
        ConvertChebyshevEval, ConvertRotateAndReduce,
        SecretGenericOpCipherPlainConversion<arith::AddFOp, ckks::AddPlainOp>,
        SecretGenericOpCipherPlainConversion<arith::AddIOp, ckks::AddPlainOp>,
        SecretGenericOpCipherPlainConversion<arith::MulFOp, ckks::MulPlainOp>,
        SecretGenericOpCipherPlainConversion<arith::MulIOp, ckks::MulPlainOp>,
        SecretGenericOpCipherPlainConversion<arith::SubFOp, ckks::SubPlainOp>,
        SecretGenericOpCipherPlainConversion<arith::SubIOp, ckks::SubPlainOp>,
        SecretGenericOpConversion<arith::NegFOp, ckks::NegateOp>,
        SecretGenericOpConversion<arith::AddFOp, ckks::AddOp>,
        SecretGenericOpConversion<arith::AddIOp, ckks::AddOp>,
        SecretGenericOpConversion<arith::MulFOp, ckks::MulOp>,
        SecretGenericOpConversion<arith::MulIOp, ckks::MulOp>,
        SecretGenericOpConversion<arith::SubFOp, ckks::SubOp>,
        SecretGenericOpConversion<arith::SubIOp, ckks::SubOp>,
        SecretGenericOpConversion<mgmt::BootstrapOp, ckks::BootstrapOp>,
        SecretGenericOpModulusSwitchConversion<ckks::RescaleOp>,
        SecretGenericOpRelinearizeConversion<ckks::RelinearizeOp>,
        SecretGenericOpRotateConversion<ckks::RotateOp>,
        SecretGenericPlaintextDivision,
        SecretGenericOpLevelReduceConversion<ckks::LevelReduceOp>>(
        typeConverter, context);

    patterns.add<ConvertClientConceal>(typeConverter, context, usePublicKey,
                                       rlweRing.value());
    patterns.add<ConvertClientReveal>(typeConverter, context, rlweRing.value());

    if (failed(applyContextAwarePartialConversion(module, target,
                                                  std::move(patterns)))) {
      return signalPassFailure();
    }

    clearAttrs(getOperation(), mgmt::MgmtDialect::kArgMgmtAttrName);
    mgmt::cleanupInitOp(getOperation());
  }
};

}  // namespace mlir::heir
