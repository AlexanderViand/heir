#include "lib/Dialect/LWE/Conversions/LWEToCheddar/LWEToCheddar.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "lib/Dialect/Cheddar/IR/CheddarDialect.h"
#include "lib/Dialect/Cheddar/IR/CheddarOps.h"
#include "lib/Dialect/Cheddar/IR/CheddarTypes.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/Orion/IR/OrionDialect.h"
#include "lib/Dialect/Orion/IR/OrionOps.h"
#include "lib/Utils/ConversionUtils.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/APInt.h"               // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"           // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Bufferization/IR/Bufferization.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

#define DEBUG_TYPE "lwe-to-cheddar"

namespace mlir::heir::lwe {

//===----------------------------------------------------------------------===//
// Type converter
//===----------------------------------------------------------------------===//
//
// The cheddar dialect is destination-passing-style on builtin tensors: a scalar
// payload value is a rank-0 `tensor<!cheddar.X>`. So a scalar `!lwe.ciphertext`
// converts to `tensor<!cheddar.ciphertext>` (rank-0), and a packed
// `tensor<Nx!lwe.ciphertext>` converts to `tensor<Nx!cheddar.ciphertext>` (a
// tensor whose ELEMENT is the scalar cheddar payload -- NOT a nested tensor, so
// the RankedTensorType rule maps payload elements directly rather than
// recursing through the scalar rule).

class ToCheddarTypeConverter : public TypeConverter {
 public:
  ToCheddarTypeConverter(MLIRContext* ctx) {
    addConversion([](Type type) { return type; });
    addConversion([ctx](lwe::LWECiphertextType type) -> Type {
      return RankedTensorType::get({}, cheddar::CiphertextType::get(ctx));
    });
    addConversion([ctx](lwe::LWEPlaintextType type) -> Type {
      return RankedTensorType::get({}, cheddar::PlaintextType::get(ctx));
    });
    // Keys are absorbed into the UserInterface (threaded as contextual args).
    addConversion([ctx](lwe::LWEPublicKeyType type) -> Type {
      return cheddar::UserInterfaceType::get(ctx);
    });
    addConversion([ctx](lwe::LWESecretKeyType type) -> Type {
      return cheddar::UserInterfaceType::get(ctx);
    });
    addConversion([this, ctx](RankedTensorType type) -> Type {
      Type elt = type.getElementType();
      // A packed payload buffer maps to a tensor of the SCALAR cheddar payload
      // (the scalar-payload rules above map the bare element type to a rank-0
      // tensor, which must not be nested inside this one).
      if (isa<lwe::LWECiphertextType>(elt))
        return RankedTensorType::get(type.getShape(),
                                     cheddar::CiphertextType::get(ctx));
      if (isa<lwe::LWEPlaintextType>(elt))
        return RankedTensorType::get(type.getShape(),
                                     cheddar::PlaintextType::get(ctx));
      return RankedTensorType::get(type.getShape(), this->convertType(elt));
    });
  }
};

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

namespace {

bool isCheddarPayload(Type t) {
  return isa<cheddar::CiphertextType, cheddar::PlaintextType,
             cheddar::ConstantType>(t);
}

// Fresh destination (DPS init) for a payload-producing cheddar op: a
// `bufferization.alloc_tensor` of the op's converted (rank-0 tensor) result.
Value makeDest(OpBuilder& b, Location loc, Type resultTy) {
  return bufferization::AllocTensorOp::create(
      b, loc, cast<RankedTensorType>(resultTy), ValueRange{});
}

template <typename... Dialects>
bool containsArgumentOfDialect(Operation* op) {
  auto funcOp = dyn_cast<func::FuncOp>(op);
  if (!funcOp) return false;
  return llvm::any_of(funcOp.getArgumentTypes(), [&](Type argType) {
    return DialectEqual<Dialects...>()(
        &getElementTypeOrSelf(argType).getDialect());
  });
}

template <typename CheddarType>
FailureOr<Value> getContextualArg(Operation* op) {
  auto result = getContextualArgFromFunc<CheddarType>(op);
  if (failed(result)) {
    return op->emitOpError()
           << "Found op in a function without a required CHEDDAR context "
              "argument. Did the AddEvaluatorArg pattern fail to run?";
  }
  return result.value();
}

FailureOr<Value> getContextualArg(Operation* op, Type type) {
  return getContextualArgFromFunc(op, type);
}

FailureOr<Value> getContextualContext(Operation* op) {
  if (auto bootCtx = getContextualArgFromFunc<cheddar::BootContextType>(op);
      succeeded(bootCtx))
    return bootCtx;
  return getContextualArg<cheddar::ContextType>(op);
}

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

// Binary ct-ct operations: ckks.add -> cheddar.add, etc.
template <typename CKKSOp, typename CheddarOp>
struct ConvertCKKSBinOp : public OpConversionPattern<CKKSOp> {
  using OpConversionPattern<CKKSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      CKKSOp op, typename CKKSOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = getContextualContext(op.getOperation());
    if (failed(ctx)) return ctx;
    Type resultTy = this->typeConverter->convertType(op.getOutput().getType());
    Value dest = makeDest(rewriter, op.getLoc(), resultTy);
    rewriter.replaceOpWithNewOp<CheddarOp>(
        op, resultTy, ctx.value(), adaptor.getLhs(), adaptor.getRhs(), dest);
    return success();
  }
};

using ConvertCKKSAddOp = ConvertCKKSBinOp<ckks::AddOp, cheddar::AddOp>;
using ConvertCKKSSubOp = ConvertCKKSBinOp<ckks::SubOp, cheddar::SubOp>;
using ConvertCKKSMulOp = ConvertCKKSBinOp<ckks::MulOp, cheddar::MultOp>;
using ConvertRAddOp = ConvertCKKSBinOp<lwe::RAddOp, cheddar::AddOp>;
using ConvertRSubOp = ConvertCKKSBinOp<lwe::RSubOp, cheddar::SubOp>;
using ConvertRMulOp = ConvertCKKSBinOp<lwe::RMulOp, cheddar::MultOp>;

// Ct-pt operations.
template <typename CKKSOp, typename CheddarOp>
struct ConvertCKKSPlainOp : public OpConversionPattern<CKKSOp> {
  using OpConversionPattern<CKKSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      CKKSOp op, typename CKKSOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = getContextualContext(op.getOperation());
    if (failed(ctx)) return ctx;

    // Ensure ciphertext is first operand (CHEDDAR convention).
    auto isCt = [](Value v) {
      auto t = dyn_cast<RankedTensorType>(v.getType());
      return t && isa<cheddar::CiphertextType>(t.getElementType());
    };
    Value ciphertext = adaptor.getLhs();
    Value plaintext = adaptor.getRhs();
    if (!isCt(ciphertext)) {
      ciphertext = adaptor.getRhs();
      plaintext = adaptor.getLhs();
    }
    Type resultTy = this->typeConverter->convertType(op.getOutput().getType());
    Value dest = makeDest(rewriter, op.getLoc(), resultTy);
    rewriter.replaceOpWithNewOp<CheddarOp>(op, resultTy, ctx.value(),
                                           ciphertext, plaintext, dest);
    return success();
  }
};

using ConvertCKKSAddPlainOp =
    ConvertCKKSPlainOp<ckks::AddPlainOp, cheddar::AddPlainOp>;
using ConvertCKKSSubPlainOp =
    ConvertCKKSPlainOp<ckks::SubPlainOp, cheddar::SubPlainOp>;
using ConvertCKKSMulPlainOp =
    ConvertCKKSPlainOp<ckks::MulPlainOp, cheddar::MultPlainOp>;
using ConvertRAddPlainOp =
    ConvertCKKSPlainOp<lwe::RAddPlainOp, cheddar::AddPlainOp>;
using ConvertRSubPlainOp =
    ConvertCKKSPlainOp<lwe::RSubPlainOp, cheddar::SubPlainOp>;
using ConvertRMulPlainOp =
    ConvertCKKSPlainOp<lwe::RMulPlainOp, cheddar::MultPlainOp>;

struct ConvertCKKSNegateOp : public OpConversionPattern<ckks::NegateOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      ckks::NegateOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = getContextualContext(op.getOperation());
    if (failed(ctx)) return ctx;
    Type resultTy = typeConverter->convertType(op.getOutput().getType());
    Value dest = makeDest(rewriter, op.getLoc(), resultTy);
    rewriter.replaceOpWithNewOp<cheddar::NegOp>(op, resultTy, ctx.value(),
                                                adaptor.getInput(), dest);
    return success();
  }
};

struct ConvertCKKSRelinOp : public OpConversionPattern<ckks::RelinearizeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      ckks::RelinearizeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = getContextualContext(op.getOperation());
    if (failed(ctx)) return ctx;
    auto multKey = getContextualArg<cheddar::EvalKeyType>(op.getOperation());
    if (failed(multKey)) return multKey;
    Type resultTy = typeConverter->convertType(op.getOutput().getType());
    Value dest = makeDest(rewriter, op.getLoc(), resultTy);
    rewriter.replaceOpWithNewOp<cheddar::RelinearizeOp>(
        op, resultTy, ctx.value(), adaptor.getInput(), multKey.value(), dest);
    return success();
  }
};

struct ConvertCKKSRescaleOp : public OpConversionPattern<ckks::RescaleOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      ckks::RescaleOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = getContextualContext(op.getOperation());
    if (failed(ctx)) return ctx;
    Type resultTy = typeConverter->convertType(op.getOutput().getType());
    Value dest = makeDest(rewriter, op.getLoc(), resultTy);
    rewriter.replaceOpWithNewOp<cheddar::RescaleOp>(op, resultTy, ctx.value(),
                                                    adaptor.getInput(), dest);
    return success();
  }
};

// Rotate -> keyless cheddar.hrot (the emitter looks up the rotation key
// inline).
struct ConvertCKKSRotateOp : public OpConversionPattern<ckks::RotateOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      ckks::RotateOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = getContextualContext(op.getOperation());
    if (failed(ctx)) return ctx;
    Value dynamicShift = adaptor.getDynamicShift();
    IntegerAttr staticShift = op.getStaticShiftAttr();
    if (!staticShift && !dynamicShift)
      return rewriter.notifyMatchFailure(
          op, "rotate op must have either static or dynamic shift");
    Type resultTy = typeConverter->convertType(op.getOutput().getType());
    Value dest = makeDest(rewriter, op.getLoc(), resultTy);
    if (dynamicShift) {
      rewriter.replaceOpWithNewOp<cheddar::HRotOp>(
          op, resultTy, ctx.value(), adaptor.getInput(), dest, dynamicShift,
          /*static_distance=*/IntegerAttr());
    } else {
      rewriter.replaceOpWithNewOp<cheddar::HRotOp>(
          op, resultTy, ctx.value(), adaptor.getInput(), dest,
          /*dynamic_distance=*/Value(), staticShift);
    }
    return success();
  }
};

struct ConvertCKKSLevelReduceOp
    : public OpConversionPattern<ckks::LevelReduceOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      ckks::LevelReduceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = getContextualContext(op.getOperation());
    if (failed(ctx)) return ctx;
    auto outputCtType = dyn_cast<lwe::LWECiphertextType>(
        getElementTypeOrSelf(op.getOutput().getType()));
    int64_t targetLevelVal =
        outputCtType ? outputCtType.getModulusChain().getCurrent() : 0;
    Type resultTy = typeConverter->convertType(op.getOutput().getType());
    Value dest = makeDest(rewriter, op.getLoc(), resultTy);
    rewriter.replaceOpWithNewOp<cheddar::LevelDownOp>(
        op, resultTy, ctx.value(), adaptor.getInput(), dest,
        rewriter.getI64IntegerAttr(targetLevelVal));
    return success();
  }
};

struct ConvertCKKSBootstrapOp : public OpConversionPattern<ckks::BootstrapOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      ckks::BootstrapOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = getContextualContext(op.getOperation());
    if (failed(ctx)) return ctx;
    auto evkMap = getContextualArg<cheddar::EvkMapType>(op.getOperation());
    if (failed(evkMap)) return evkMap;
    Type resultTy = typeConverter->convertType(op.getOutput().getType());
    Value dest = makeDest(rewriter, op.getLoc(), resultTy);
    rewriter.replaceOpWithNewOp<cheddar::BootOp>(
        op, resultTy, ctx.value(), adaptor.getInput(), evkMap.value(), dest);
    return success();
  }
};

// Encode (forwards level + nominal scale; the emitter re-derives the canonical
// per-level scale).
struct ConvertLWEEncodeOp : public OpConversionPattern<lwe::RLWEEncodeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      lwe::RLWEEncodeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto encoder = getContextualArg<cheddar::EncoderType>(op.getOperation());
    if (failed(encoder)) return encoder;
    auto ptType = dyn_cast<lwe::LWEPlaintextType>(op.getOutput().getType());
    auto invEncoding = ptType ? dyn_cast<lwe::InverseCanonicalEncodingAttr>(
                                    ptType.getPlaintextSpace().getEncoding())
                              : nullptr;
    if (!invEncoding)
      return op.emitOpError()
             << "cannot lower to cheddar.encode: plaintext has no "
                "InverseCanonicalEncoding (not a CKKS plaintext)";
    double scale = static_cast<double>(invEncoding.getScalingFactor());
    int64_t level = op.getLevel() ? op.getLevel().value() : 0;
    Type resultTy = typeConverter->convertType(op.getOutput().getType());
    Value dest = makeDest(rewriter, op.getLoc(), resultTy);
    rewriter.replaceOpWithNewOp<cheddar::EncodeOp>(
        op, resultTy, encoder.value(), adaptor.getInput(), dest,
        rewriter.getI64IntegerAttr(level), rewriter.getF64FloatAttr(scale));
    return success();
  }
};

struct ConvertLWEDecryptOp : public OpConversionPattern<lwe::RLWEDecryptOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      lwe::RLWEDecryptOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ui = getContextualArg<cheddar::UserInterfaceType>(op.getOperation());
    if (failed(ui)) return ui;
    Type resultTy = typeConverter->convertType(op.getOutput().getType());
    Value dest = makeDest(rewriter, op.getLoc(), resultTy);
    rewriter.replaceOpWithNewOp<cheddar::DecryptOp>(op, resultTy, ui.value(),
                                                    adaptor.getInput(), dest);
    return success();
  }
};

struct ConvertLWEEncryptOp : public OpConversionPattern<lwe::RLWEEncryptOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      lwe::RLWEEncryptOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ui = getContextualArg<cheddar::UserInterfaceType>(op.getOperation());
    if (failed(ui)) return ui;
    Type resultTy = typeConverter->convertType(op.getOutput().getType());
    Value dest = makeDest(rewriter, op.getLoc(), resultTy);
    rewriter.replaceOpWithNewOp<cheddar::EncryptOp>(op, resultTy, ui.value(),
                                                    adaptor.getInput(), dest);
    return success();
  }
};

// Decode is already destination-passing on the float `value` buffer.
struct ConvertLWEDecodeOp : public OpConversionPattern<lwe::RLWEDecodeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      lwe::RLWEDecodeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto encoder = getContextualArg<cheddar::EncoderType>(op.getOperation());
    if (failed(encoder)) return encoder;
    auto outTy = cast<RankedTensorType>(op.getOutput().getType());
    Value dest = tensor::EmptyOp::create(
        rewriter, op.getLoc(), outTy.getShape(), outTy.getElementType());
    rewriter.replaceOpWithNewOp<cheddar::DecodeOp>(op, outTy, encoder.value(),
                                                   adaptor.getInput(), dest);
    return success();
  }
};

// orion.linear_transform -> cheddar.linear_transform.
struct ConvertOrionLinearTransformOp
    : public OpConversionPattern<orion::LinearTransformOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      orion::LinearTransformOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = getContextualContext(op.getOperation());
    if (failed(ctx)) return ctx;
    auto evkMap = getContextualArg<cheddar::EvkMapType>(op.getOperation());
    if (failed(evkMap)) return evkMap;
    auto level = rewriter.getI64IntegerAttr(op.getOrionLevelAttr().getInt());
    double bsgsRatio = op.getBsgsRatioAttr().getValueAsDouble();
    int64_t ratio =
        bsgsRatio > 0 ? static_cast<int64_t>(std::llround(bsgsRatio)) : 2;
    if (ratio < 1) ratio = 1;
    int64_t maxRot = 0;
    for (int32_t d : op.getDiagonalIndicesAttr().asArrayRef())
      maxRot = std::max<int64_t>(maxRot, d);
    int64_t need = maxRot + 1;
    int64_t gs = static_cast<int64_t>(
        std::ceil(std::sqrt(static_cast<double>(need) / ratio)));
    if (gs < 1) gs = 1;
    int64_t bs = (need + gs - 1) / gs;
    Type resultTy = typeConverter->convertType(op.getResult().getType());
    Value dest = makeDest(rewriter, op.getLoc(), resultTy);
    rewriter.replaceOpWithNewOp<cheddar::LinearTransformOp>(
        op, resultTy, ctx.value(), adaptor.getInput(), evkMap.value(),
        adaptor.getDiagonals(), dest, op.getDiagonalIndicesAttr(), level,
        rewriter.getI64IntegerAttr(bs), rewriter.getI64IntegerAttr(gs));
    return success();
  }
};

// orion.chebyshev -> cheddar.eval_poly.
struct ConvertOrionChebyshevOp
    : public OpConversionPattern<orion::ChebyshevOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      orion::ChebyshevOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = getContextualContext(op.getOperation());
    if (failed(ctx)) return ctx;
    auto evkMap = getContextualArg<cheddar::EvkMapType>(op.getOperation());
    if (failed(evkMap)) return evkMap;
    double domainStart = op.getDomainStartAttr().getValueAsDouble();
    double domainEnd = op.getDomainEndAttr().getValueAsDouble();
    if (domainStart != -1.0 || domainEnd != 1.0)
      return op.emitError()
             << "cannot lower orion.chebyshev with approximation domain ["
             << domainStart << ", " << domainEnd
             << "] to the cheddar backend: CHEDDAR's EvalPoly only evaluates "
                "on [-1, 1]. Normalize the domain before lowering.";
    auto inTy = dyn_cast<lwe::LWECiphertextType>(op.getInput().getType());
    auto resTy = dyn_cast<lwe::LWECiphertextType>(op.getResult().getType());
    if (!inTy || !inTy.getModulusChain() || !resTy || !resTy.getModulusChain())
      return op.emitOpError()
             << "cannot lower to cheddar.eval_poly: input/result ciphertext "
                "has no modulus chain (run the upstream CKKS level analysis "
                "first)";
    int64_t level = inTy.getModulusChain().getCurrent();
    int64_t outputLevel = resTy.getModulusChain().getCurrent();
    Type resultTy = typeConverter->convertType(op.getResult().getType());
    Value dest = makeDest(rewriter, op.getLoc(), resultTy);
    rewriter.replaceOpWithNewOp<cheddar::EvalPolyOp>(
        op, resultTy, ctx.value(), adaptor.getInput(), evkMap.value(), dest,
        op.getCoefficientsAttr(), rewriter.getI64IntegerAttr(level),
        rewriter.getI64IntegerAttr(outputLevel));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Payload packing: scalar-index tensor ops -> rank-reducing slice ops
//===----------------------------------------------------------------------===//
//
// In DPS form a "scalar" payload is a rank-0 `tensor<!cheddar.X>`, so the
// source's scalar packing ops (which produce / consume the bare payload element
// of a `tensor<Nx!lwe.X>`) must become rank-reducing slice ops:
//   tensor.extract %v[i]          -> tensor.extract_slice %v[i][1][1] : ->
//   tensor<!X> tensor.insert  %s into %v[i]  -> tensor.insert_slice  %s into
//   %v[i][1][1] tensor.from_elements %s0,..   -> tensor.empty + insert_slice
//   per element

static void unitSlice(OpBuilder& b, ValueRange indices,
                      SmallVector<OpFoldResult>& offsets,
                      SmallVector<OpFoldResult>& sizes,
                      SmallVector<OpFoldResult>& strides) {
  for (Value idx : indices) {
    offsets.push_back(idx);
    sizes.push_back(b.getIndexAttr(1));
    strides.push_back(b.getIndexAttr(1));
  }
}

struct ConvertPayloadExtract : public OpConversionPattern<tensor::ExtractOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      tensor::ExtractOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto srcTy = dyn_cast<RankedTensorType>(adaptor.getTensor().getType());
    if (!srcTy || !isCheddarPayload(srcTy.getElementType())) return failure();
    auto resTy = RankedTensorType::get({}, srcTy.getElementType());
    SmallVector<OpFoldResult> offsets, sizes, strides;
    unitSlice(rewriter, adaptor.getIndices(), offsets, sizes, strides);
    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        op, resTy, adaptor.getTensor(), offsets, sizes, strides);
    return success();
  }
};

struct ConvertPayloadInsert : public OpConversionPattern<tensor::InsertOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      tensor::InsertOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto destTy = dyn_cast<RankedTensorType>(adaptor.getDest().getType());
    if (!destTy || !isCheddarPayload(destTy.getElementType())) return failure();
    SmallVector<OpFoldResult> offsets, sizes, strides;
    unitSlice(rewriter, adaptor.getIndices(), offsets, sizes, strides);
    rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
        op, adaptor.getScalar(), adaptor.getDest(), offsets, sizes, strides);
    return success();
  }
};

struct ConvertPayloadFromElements
    : public OpConversionPattern<tensor::FromElementsOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      tensor::FromElementsOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto resTy =
        dyn_cast<RankedTensorType>(typeConverter->convertType(op.getType()));
    if (!resTy || !isCheddarPayload(resTy.getElementType())) return failure();
    if (resTy.getRank() != 1 ||
        resTy.getDimSize(0) != (int64_t)adaptor.getElements().size())
      return rewriter.notifyMatchFailure(op, "unsupported from_elements shape");
    Value acc = tensor::EmptyOp::create(rewriter, op.getLoc(), resTy.getShape(),
                                        resTy.getElementType());
    SmallVector<OpFoldResult> sizes{rewriter.getIndexAttr(1)};
    SmallVector<OpFoldResult> strides{rewriter.getIndexAttr(1)};
    for (auto [i, elt] : llvm::enumerate(adaptor.getElements())) {
      SmallVector<OpFoldResult> offsets{rewriter.getIndexAttr((int64_t)i)};
      acc = tensor::InsertSliceOp::create(rewriter, op.getLoc(), elt, acc,
                                          offsets, sizes, strides);
    }
    rewriter.replaceOp(op, acc);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Context-argument threading (mirrors LWEToLattigo)
//===----------------------------------------------------------------------===//

struct AddCheddarContextArg : public OpConversionPattern<func::FuncOp> {
  AddCheddarContextArg(
      mlir::MLIRContext* context,
      const std::vector<std::pair<Type, OpPredicate>>& evaluators)
      : OpConversionPattern<func::FuncOp>(context, /* benefit= */ 2),
        evaluators(evaluators) {}
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::FuncOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    SmallVector<Type, 4> selectedTypes;
    for (const auto& evaluator : evaluators)
      if (evaluator.second(op)) selectedTypes.push_back(evaluator.first);
    if (selectedTypes.empty())
      return rewriter.notifyMatchFailure(op, "no CHEDDAR context needed");
    SmallVector<DictionaryAttr> argAttrs(selectedTypes.size(), nullptr);
    SmallVector<Location> argLocs(selectedTypes.size(), op.getLoc());
    rewriter.modifyOpInPlace(op, [&] {
      SmallVector<unsigned> indices(selectedTypes.size(), 0);
      (void)op.insertArguments(indices, selectedTypes, argAttrs, argLocs);
    });
    return success();
  }

 private:
  std::vector<std::pair<Type, OpPredicate>> evaluators;
};

struct ConvertCheddarFuncCallOp : public OpConversionPattern<func::CallOp> {
  ConvertCheddarFuncCallOp(
      mlir::MLIRContext* context,
      const std::vector<std::pair<Type, OpPredicate>>& evaluators)
      : OpConversionPattern<func::CallOp>(context), evaluators(evaluators) {}
  using OpConversionPattern<func::CallOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::CallOp op, typename func::CallOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto funcOp = getCalledFunction(op);
    if (failed(funcOp))
      return rewriter.notifyMatchFailure(op, "could not find callee function");
    SmallVector<Value> selectedValues;
    for (const auto& evaluator : evaluators) {
      auto result = getContextualArg(op.getOperation(), evaluator.first);
      if (failed(result)) continue;
      if (!llvm::any_of(funcOp.value().getArgumentTypes(), [&](Type argType) {
            return evaluator.first == argType;
          }))
        continue;
      selectedValues.push_back(result.value());
    }
    SmallVector<Value> newOperands;
    for (auto v : selectedValues) newOperands.push_back(v);
    for (auto operand : adaptor.getOperands()) newOperands.push_back(operand);
    SmallVector<NamedAttribute> dialectAttrs(op->getDialectAttrs());
    rewriter
        .replaceOpWithNewOp<func::CallOp>(op, op.getCallee(),
                                          op.getResultTypes(), newOperands)
        ->setDialectAttrs(dialectAttrs);
    return success();
  }

 private:
  std::vector<std::pair<Type, OpPredicate>> evaluators;
};

}  // namespace

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

#define GEN_PASS_DEF_LWETOCHEDDAR
#include "lib/Dialect/LWE/Conversions/LWEToCheddar/LWEToCheddar.h.inc"

namespace {

struct LWEToCheddar : public impl::LWEToCheddarBase<LWEToCheddar> {
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();
    ToCheddarTypeConverter typeConverter(context);

    if (!moduleIsCKKS(module)) {
      module->emitOpError("CHEDDAR backend only supports CKKS scheme");
      return signalPassFailure();
    }

    ConversionTarget target(*context);
    target.addLegalDialect<cheddar::CheddarDialect>();
    target.addLegalDialect<bufferization::BufferizationDialect>();
    target.addIllegalDialect<ckks::CKKSDialect, orion::OrionDialect>();
    target
        .addIllegalOp<lwe::RLWEEncryptOp, lwe::RLWEDecryptOp, lwe::RLWEEncodeOp,
                      lwe::RLWEDecodeOp, lwe::RAddOp, lwe::RSubOp, lwe::RMulOp,
                      lwe::RAddPlainOp, lwe::RSubPlainOp, lwe::RMulPlainOp>();

    RewritePatternSet patterns(context);
    addStructuralConversionPatterns(typeConverter, patterns, target);
    addTensorConversionPatterns(typeConverter, patterns, target);

    auto hasCryptoOps = [&](Operation* op) -> bool {
      return containsArgumentOfDialect<lwe::LWEDialect, ckks::CKKSDialect>(op);
    };
    auto hasEncodeOps = [&](Operation* op) -> bool {
      auto funcOp = dyn_cast<func::FuncOp>(op);
      if (!funcOp) return false;
      bool found = false;
      funcOp->walk([&](lwe::RLWEEncodeOp) { found = true; });
      return found;
    };
    auto needsEvkMap = [&](Operation* op) -> bool {
      auto funcOp = dyn_cast<func::FuncOp>(op);
      if (!funcOp) return false;
      bool found = false;
      funcOp->walk([&](Operation* inner) {
        if (isa<ckks::BootstrapOp, orion::LinearTransformOp,
                orion::ChebyshevOp>(inner))
          found = true;
      });
      return found;
    };
    auto funcBootstraps = [&](Operation* op) -> bool {
      auto funcOp = dyn_cast<func::FuncOp>(op);
      if (!funcOp) return false;
      bool found = false;
      funcOp->walk([&](ckks::BootstrapOp) { found = true; });
      return found;
    };
    auto hasCryptoNoBoot = [&](Operation* op) -> bool {
      return hasCryptoOps(op) && !funcBootstraps(op);
    };
    auto hasCryptoOrEncode = [&](Operation* op) {
      return hasCryptoOps(op) || hasEncodeOps(op);
    };
    std::vector<std::pair<Type, OpPredicate>> evaluators = {
        {cheddar::ContextType::get(context), hasCryptoNoBoot},
        {cheddar::BootContextType::get(context), funcBootstraps},
        {cheddar::EncoderType::get(context), hasCryptoOrEncode},
        {cheddar::UserInterfaceType::get(context), hasCryptoOps},
        {cheddar::EvalKeyType::get(context), hasCryptoOps},
        {cheddar::EvkMapType::get(context), needsEvkMap},
    };

    patterns.add<AddCheddarContextArg>(context, evaluators);
    patterns.add<ConvertCheddarFuncCallOp>(context, evaluators);

    patterns.add<ConvertCKKSAddOp, ConvertCKKSSubOp, ConvertCKKSMulOp,
                 ConvertCKKSAddPlainOp, ConvertCKKSSubPlainOp,
                 ConvertCKKSMulPlainOp, ConvertCKKSNegateOp, ConvertCKKSRelinOp,
                 ConvertCKKSRescaleOp, ConvertCKKSRotateOp,
                 ConvertCKKSLevelReduceOp, ConvertCKKSBootstrapOp>(
        typeConverter, context);
    patterns.add<ConvertRAddOp, ConvertRSubOp, ConvertRMulOp,
                 ConvertRAddPlainOp, ConvertRSubPlainOp, ConvertRMulPlainOp>(
        typeConverter, context);
    patterns.add<ConvertLWEEncodeOp, ConvertLWEDecodeOp, ConvertLWEEncryptOp,
                 ConvertLWEDecryptOp>(typeConverter, context);
    patterns.add<ConvertOrionLinearTransformOp, ConvertOrionChebyshevOp>(
        typeConverter, context);
    // Payload packing ops -> rank-reducing slice ops (benefit 2 so they win
    // over the structural tensor conversion for payload-typed tensors).
    patterns.add<ConvertPayloadExtract, ConvertPayloadInsert,
                 ConvertPayloadFromElements>(typeConverter, context,
                                             /*benefit=*/2);

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      bool hasCheddarCtxArg =
          op.getFunctionType().getNumInputs() > 0 &&
          containsArgumentOfType<cheddar::ContextType, cheddar::BootContextType,
                                 cheddar::EncoderType,
                                 cheddar::UserInterfaceType,
                                 cheddar::EvalKeyType>(op);
      bool hasCryptoArg =
          containsArgumentOfDialect<lwe::LWEDialect, ckks::CKKSDialect>(op);
      bool hasEncodeOp = false;
      op.walk([&](lwe::RLWEEncodeOp) { hasEncodeOp = true; });
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody()) &&
             (!(hasCryptoArg || hasEncodeOp) || hasCheddarCtxArg);
    });

    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      auto operandTypes = op.getCalleeType().getInputs();
      auto containsCryptoArg = llvm::any_of(operandTypes, [&](Type argType) {
        return DialectEqual<lwe::LWEDialect, ckks::CKKSDialect>()(
            &argType.getDialect());
      });
      auto hasCheddarCtxArg =
          !operandTypes.empty() &&
          mlir::isa<cheddar::ContextType, cheddar::BootContextType>(
              *operandTypes.begin());
      bool signatureConsistent = false;
      FailureOr<func::FuncOp> callee = getCalledFunction(op);
      if (succeeded(callee))
        signatureConsistent =
            callee.value().getFunctionType() == op.getCalleeType();
      return (!containsCryptoArg || hasCheddarCtxArg) && signatureConsistent;
    });

    target.markUnknownOpDynamicallyLegal(
        [&](Operation* op) -> std::optional<bool> {
          return typeConverter.isLegal(op);
        });

    ConversionConfig config;
    config.allowPatternRollback = false;
    if (failed(applyPartialConversion(module, target, std::move(patterns),
                                      config))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace mlir::heir::lwe
