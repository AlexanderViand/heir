#include "lib/Dialect/LWE/Conversions/LWEToCheddar/LWEToCheddar.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "lib/Dialect/Cheddar/IR/CheddarDialect.h"
#include "lib/Dialect/Cheddar/IR/CheddarOps.h"
#include "lib/Dialect/Cheddar/IR/CheddarTypes.h"
#include "lib/Dialect/Kernel/IR/KernelOps.h"
#include "lib/Dialect/Kernel/IR/KernelTypes.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/Preprocessing/Conversions/Util.h"
#include "lib/Dialect/Preprocessing/IR/PreprocessingDialect.h"
#include "lib/Dialect/Preprocessing/IR/PreprocessingTypes.h"
#include "lib/Utils/ConversionUtils.h"
#include "lib/Utils/RotationUtils.h"
#include "lib/Utils/TargetUtils.h"
#include "lib/Utils/Utils.h"
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
    addConversion([ctx](kernel::PreparedLinearTransformType type) -> Type {
      return RankedTensorType::get({}, cheddar::LinearTransformType::get(ctx));
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
    // split-preprocessing storage: convert its plaintext element types
    // (lwe.lwe_plaintext -> rank-0 tensor<!cheddar.plaintext>); the storage
    // itself is lowered to memref later by --preprocessing-to-cheddar.
    addConversion([this](preprocessing::PreprocessingStorageType type) -> Type {
      return preprocessing::convertStorageElementTypes(type, this);
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

// A shape-only destination for operations without a reusable payload input.
// One-Shot Bufferize chooses the eventual allocation.
Value makeEmptyDest(OpBuilder& b, Location loc, Type resultTy) {
  auto tensorType = cast<RankedTensorType>(resultTy);
  return tensor::EmptyOp::create(b, loc, tensorType.getShape(),
                                 tensorType.getElementType());
}

// Reuse only a payload produced locally by another Cheddar operation. All
// scale-snu APIs represented by Cheddar DPS ops accept an explicitly identical
// input and output object, but a function argument or preprocessing-storage
// view is borrowed by the generated C++ ABI and must not become mutable merely
// because it happens to die in MLIR SSA. One-Shot Bufferize still resolves
// later-use conflicts for eligible local values by allocating an uninitialized
// buffer for the fully-overwriting op.
Value makeReusableDest(OpBuilder& b, Location loc, Type resultTy,
                       Value candidate) {
  Operation* definingOp = candidate.getDefiningOp();
  if (definingOp && definingOp->getDialect() &&
      definingOp->getDialect()->getNamespace() ==
          cheddar::CheddarDialect::getDialectNamespace())
    return candidate;
  return makeEmptyDest(b, loc, resultTy);
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
    // CHEDDAR's element-wise arithmetic APIs can overwrite their lhs. This is
    // only a DPS destination hint: One-Shot Bufferize still allocates a fresh
    // output if lhs remains live.
    Value dest =
        makeReusableDest(rewriter, op.getLoc(), resultTy, adaptor.getLhs());
    auto result =
        CheddarOp::create(rewriter, op.getLoc(), resultTy, ctx.value(),
                          adaptor.getLhs(), adaptor.getRhs(), dest);
    rewriter.replaceOp(op, result);
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
    // These APIs can overwrite the ciphertext operand. One-Shot Bufferize
    // decides whether doing so is valid for this particular SSA use-def chain.
    Value dest = makeReusableDest(rewriter, op.getLoc(), resultTy, ciphertext);
    auto result = CheddarOp::create(rewriter, op.getLoc(), resultTy,
                                    ctx.value(), ciphertext, plaintext, dest);
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Ct-pt subtraction. cheddar.sub_plain computes ct - pt, so it requires the
// ciphertext first -- but unlike add/mul, subtraction is NOT commutative, so a
// blind operand swap turns `pt - ct` into `ct - pt` (a sign flip). When the
// plaintext is the lhs, lower `pt - ct` as `(-ct) + pt` (negate then
// add_plain). Mirrors LWEToLattigo's ConvertRlweSubPlainOp.
template <typename CKKSOp>
struct ConvertCKKSSubPlainOpImpl : public OpConversionPattern<CKKSOp> {
  using OpConversionPattern<CKKSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      CKKSOp op, typename CKKSOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = getContextualContext(op.getOperation());
    if (failed(ctx)) return ctx;
    auto isCt = [](Value v) {
      auto t = dyn_cast<RankedTensorType>(v.getType());
      return t && isa<cheddar::CiphertextType>(t.getElementType());
    };
    Type resultTy = this->typeConverter->convertType(op.getOutput().getType());
    if (isCt(adaptor.getLhs())) {
      // ct - pt: direct.
      Value dest =
          makeReusableDest(rewriter, op.getLoc(), resultTy, adaptor.getLhs());
      auto result = cheddar::SubPlainOp::create(rewriter, op.getLoc(), resultTy,
                                                ctx.value(), adaptor.getLhs(),
                                                adaptor.getRhs(), dest);
      rewriter.replaceOp(op, result);
      return success();
    }
    // pt - ct  ==  (-ct) + pt
    Value plaintext = adaptor.getLhs();
    Value ciphertext = adaptor.getRhs();
    Value negDest =
        makeReusableDest(rewriter, op.getLoc(), resultTy, ciphertext);
    Value negated = cheddar::NegOp::create(rewriter, op.getLoc(), resultTy,
                                           ctx.value(), ciphertext, negDest)
                        ->getResult(0);
    Value addDest = negated;
    auto result =
        cheddar::AddPlainOp::create(rewriter, op.getLoc(), resultTy,
                                    ctx.value(), negated, plaintext, addDest);
    rewriter.replaceOp(op, result);
    return success();
  }
};

using ConvertCKKSAddPlainOp =
    ConvertCKKSPlainOp<ckks::AddPlainOp, cheddar::AddPlainOp>;
using ConvertCKKSSubPlainOp = ConvertCKKSSubPlainOpImpl<ckks::SubPlainOp>;
using ConvertCKKSMulPlainOp =
    ConvertCKKSPlainOp<ckks::MulPlainOp, cheddar::MultPlainOp>;
using ConvertRAddPlainOp =
    ConvertCKKSPlainOp<lwe::RAddPlainOp, cheddar::AddPlainOp>;
using ConvertRSubPlainOp = ConvertCKKSSubPlainOpImpl<lwe::RSubPlainOp>;
using ConvertRMulPlainOp =
    ConvertCKKSPlainOp<lwe::RMulPlainOp, cheddar::MultPlainOp>;

template <typename SourceOp>
struct ConvertNegateOp : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      SourceOp op, typename SourceOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = getContextualContext(op.getOperation());
    if (failed(ctx)) return ctx;
    Type resultTy = this->typeConverter->convertType(op.getOutput().getType());
    Value dest =
        makeReusableDest(rewriter, op.getLoc(), resultTy, adaptor.getInput());
    auto negated = cheddar::NegOp::create(
        rewriter, op.getLoc(), resultTy, ctx.value(), adaptor.getInput(), dest);
    rewriter.replaceOp(op, negated);
    return success();
  }
};

using ConvertCKKSNegateOp = ConvertNegateOp<ckks::NegateOp>;
using ConvertRNegateOp = ConvertNegateOp<lwe::RNegateOp>;

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
    Value dest =
        makeReusableDest(rewriter, op.getLoc(), resultTy, adaptor.getInput());
    auto result = cheddar::RelinearizeOp::create(
        rewriter, op.getLoc(), resultTy, ctx.value(), adaptor.getInput(),
        multKey.value(), dest);
    rewriter.replaceOp(op, result);
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
    Value dest =
        makeReusableDest(rewriter, op.getLoc(), resultTy, adaptor.getInput());
    auto result = cheddar::RescaleOp::create(
        rewriter, op.getLoc(), resultTy, ctx.value(), adaptor.getInput(), dest);
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Rotate -> cheddar.hrot; the rotation key is looked up from the function's
// UserInterface operand by the emitter.
struct ConvertCKKSRotateOp : public OpConversionPattern<ckks::RotateOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      ckks::RotateOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = getContextualContext(op.getOperation());
    if (failed(ctx)) return ctx;
    auto ui = getContextualArg<cheddar::UserInterfaceType>(op.getOperation());
    if (failed(ui)) return ui;
    Value dynamicShift = adaptor.getDynamicShift();
    IntegerAttr staticShift = op.getStaticShiftAttr();
    if (!staticShift && !dynamicShift)
      return rewriter.notifyMatchFailure(
          op, "rotate op must have either static or dynamic shift");
    Type resultTy = typeConverter->convertType(op.getOutput().getType());
    auto inputType = dyn_cast<lwe::LWECiphertextType>(
        getElementTypeOrSelf(op.getInput().getType()));
    if (!inputType || !inputType.getModulusChain())
      return rewriter.notifyMatchFailure(
          op, "cannot determine the ciphertext level for rotation");
    IntegerAttr level =
        rewriter.getI64IntegerAttr(inputType.getModulusChain().getCurrent());
    if (staticShift) {
      int64_t ringDegree = inputType.getPlaintextSpace()
                               .getRing()
                               .getPolynomialModulus()
                               .getPolynomial()
                               .getDegree();
      int64_t distance =
          normalizeRotation(staticShift.getInt(), ringDegree / 2);
      if (distance == 0) {
        rewriter.replaceOp(op, adaptor.getInput());
        return success();
      }
      staticShift = rewriter.getI64IntegerAttr(distance);
    }
    Value dest =
        makeReusableDest(rewriter, op.getLoc(), resultTy, adaptor.getInput());
    if (dynamicShift) {
      auto result = cheddar::HRotOp::create(
          rewriter, op.getLoc(), resultTy, ctx.value(), ui.value(),
          adaptor.getInput(), dest, dynamicShift,
          /*static_distance=*/IntegerAttr(), level);
      rewriter.replaceOp(op, result);
    } else {
      auto result = cheddar::HRotOp::create(
          rewriter, op.getLoc(), resultTy, ctx.value(), ui.value(),
          adaptor.getInput(), dest,
          /*dynamic_distance=*/Value(), staticShift, level);
      rewriter.replaceOp(op, result);
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
    if (!outputCtType || !outputCtType.getModulusChain())
      return op.emitOpError(
          "cannot lower level_reduce without an output modulus chain");
    int64_t targetLevelVal = outputCtType.getModulusChain().getCurrent();
    Type resultTy = typeConverter->convertType(op.getOutput().getType());
    Value dest =
        makeReusableDest(rewriter, op.getLoc(), resultTy, adaptor.getInput());
    auto result = cheddar::LevelDownOp::create(
        rewriter, op.getLoc(), resultTy, ctx.value(), adaptor.getInput(), dest,
        rewriter.getI64IntegerAttr(targetLevelVal));
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertCKKSBootstrapOp : public OpConversionPattern<ckks::BootstrapOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      ckks::BootstrapOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    if (op.getTargetLevel())
      return rewriter.notifyMatchFailure(
          op,
          "bootstrap target levels must be resolved before CHEDDAR lowering");
    auto ctx = getContextualContext(op.getOperation());
    if (failed(ctx)) return ctx;
    auto evkMap = getContextualArg<cheddar::EvkMapType>(op.getOperation());
    if (failed(evkMap)) return evkMap;
    Type resultTy = typeConverter->convertType(op.getOutput().getType());
    Value dest =
        makeReusableDest(rewriter, op.getLoc(), resultTy, adaptor.getInput());
    auto result =
        cheddar::BootOp::create(rewriter, op.getLoc(), resultTy, ctx.value(),
                                adaptor.getInput(), evkMap.value(), dest);
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Encode at the level and logarithmic scale chosen by the upstream CKKS scale
// management pipeline. CHEDDAR accepts the corresponding linear scale.
struct ConvertLWEEncodeOp : public OpConversionPattern<lwe::RLWEEncodeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      lwe::RLWEEncodeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto encoder = getContextualArg<cheddar::EncoderType>(op.getOperation());
    if (failed(encoder)) return encoder;
    if (!op.getLevel())
      return op.emitOpError()
             << "cannot lower to cheddar.encode without an explicit level";
    int64_t level = op.getLevel().value();
    Type resultTy = typeConverter->convertType(op.getOutput().getType());
    Value dest = makeEmptyDest(rewriter, op.getLoc(), resultTy);
    auto result = cheddar::EncodeOp::create(
        rewriter, op.getLoc(), resultTy, encoder.value(), adaptor.getInput(),
        dest, rewriter.getI64IntegerAttr(level), op.getScaleAttr());
    rewriter.replaceOp(op, result);
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
    Value dest = makeEmptyDest(rewriter, op.getLoc(), resultTy);
    auto result = cheddar::DecryptOp::create(
        rewriter, op.getLoc(), resultTy, ui.value(), adaptor.getInput(), dest);
    rewriter.replaceOp(op, result);
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
    Value dest = makeEmptyDest(rewriter, op.getLoc(), resultTy);
    auto result = cheddar::EncryptOp::create(
        rewriter, op.getLoc(), resultTy, ui.value(), adaptor.getInput(), dest);
    rewriter.replaceOp(op, result);
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
    auto result =
        cheddar::DecodeOp::create(rewriter, op.getLoc(), outTy, encoder.value(),
                                  adaptor.getInput(), dest);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct LinearTransformPlan {
  DenseI32ArrayAttr diagonalIndices;
  int64_t width;
  int64_t bs;
  int64_t gs;
  bool minKs;
};

DenseI32ArrayAttr convertI64ArrayAttr(PatternRewriter& rewriter,
                                      DenseI64ArrayAttr attr) {
  if (!attr) return nullptr;
  SmallVector<int32_t> values;
  values.reserve(attr.size());
  for (int64_t value : attr.asArrayRef())
    values.push_back(static_cast<int32_t>(value));
  return rewriter.getDenseI32ArrayAttr(values);
}

FailureOr<LinearTransformPlan> getLinearTransformPlan(
    Operation* op, int64_t width, ArrayRef<int64_t> diagonalIndices,
    double ratio, bool enableMinKs, PatternRewriter& rewriter) {
  if (width <= 0)
    return op->emitOpError("requires a statically known positive width");
  if (!std::isfinite(ratio) || ratio <= 0)
    return op->emitOpError("bsgs_ratio must be finite and positive");

  SmallVector<int32_t> indices;
  indices.reserve(diagonalIndices.size());
  int64_t stride = 0;
  int64_t maxRotation = 0;
  for (int64_t index : diagonalIndices) {
    int64_t normalized = ((index % width) + width) % width;
    if (normalized > std::numeric_limits<int32_t>::max())
      return op->emitOpError("normalized diagonal index does not fit i32: ")
             << normalized;
    indices.push_back(static_cast<int32_t>(normalized));
    stride = std::gcd(stride, normalized);
    maxRotation = std::max(maxRotation, normalized);
  }
  if (indices.size() < 2 || stride == 0)
    return op->emitOpError(
        "scale-snu CHEDDAR linear transforms require at least two "
        "diagonals and one non-zero rotation");

  int64_t steps = maxRotation / stride + 1;
  int64_t gs = std::max<int64_t>(
      1, static_cast<int64_t>(std::ceil(std::sqrt(steps / ratio))));
  int64_t bs = (steps + gs - 1) / gs;
  auto indicesAttr = rewriter.getDenseI32ArrayAttr(indices);
  return LinearTransformPlan{
      indicesAttr, width, bs, gs,
      enableMinKs && cheddar::supportsMinKs(indicesAttr, width, bs, gs)};
}

FailureOr<LinearTransformPlan> getLinearTransformPlan(
    Operation* op, Type diagonalsType, ArrayRef<int64_t> diagonalIndices,
    double ratio, bool enableMinKs, PatternRewriter& rewriter) {
  auto shapedType = dyn_cast<ShapedType>(diagonalsType);
  if (!shapedType || !shapedType.hasRank() || shapedType.getRank() != 2)
    return op->emitOpError("requires statically shaped 2D diagonals");
  return getLinearTransformPlan(op, shapedType.getDimSize(1), diagonalIndices,
                                ratio, enableMinKs, rewriter);
}

// kernel.linear_transform -> cheddar.linear_transform.
struct ConvertKernelLinearTransformOp
    : public OpConversionPattern<kernel::LinearTransformOp> {
  ConvertKernelLinearTransformOp(const TypeConverter& converter,
                                 MLIRContext* context, bool enableMinKs)
      : OpConversionPattern(converter, context), enableMinKs(enableMinKs) {}

  LogicalResult matchAndRewrite(
      kernel::LinearTransformOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = getContextualContext(op.getOperation());
    if (failed(ctx)) return ctx;
    auto evkMap = getContextualArg<cheddar::EvkMapType>(op.getOperation());
    if (failed(evkMap)) return evkMap;

    auto inputType = dyn_cast<lwe::LWECiphertextType>(
        getElementTypeOrSelf(op.getInput().getType()));
    auto outputType = dyn_cast<lwe::LWECiphertextType>(
        getElementTypeOrSelf(op.getOutput().getType()));
    if (!inputType || !inputType.getModulusChain() || !outputType ||
        !outputType.getModulusChain())
      return op.emitOpError(
          "cannot lower to cheddar.linear_transform without input and output "
          "modulus chains; run CKKS level analysis first");
    int64_t inputLevel = inputType.getModulusChain().getCurrent();
    int64_t outputLevel = outputType.getModulusChain().getCurrent();
    if (inputLevel <= 0)
      return op.emitOpError(
          "scale-snu CHEDDAR linear transforms require an input level above "
          "zero");
    if (outputLevel != inputLevel - 1)
      return op.emitOpError(
                 "scale-snu CHEDDAR linear transforms consume exactly one "
                 "level; expected output level ")
             << inputLevel - 1 << " but got " << outputLevel;

    double ratio = 4.0;
    if (auto ratioAttr = op.getBsgsRatioAttr())
      ratio = ratioAttr.getValueAsDouble();
    auto plan = getLinearTransformPlan(op, op.getDiagonals().getType(),
                                       op.getDiagonalIndices(), ratio,
                                       enableMinKs, rewriter);
    if (failed(plan)) return failure();

    Type resultType = typeConverter->convertType(op.getOutput().getType());
    Value dest =
        makeReusableDest(rewriter, op.getLoc(), resultType, adaptor.getInput());
    auto result = cheddar::LinearTransformOp::create(
        rewriter, op.getLoc(), resultType, ctx.value(), adaptor.getInput(),
        evkMap.value(), adaptor.getDiagonals(), dest, plan->diagonalIndices,
        convertI64ArrayAttr(rewriter, op.getSourceRowIndicesAttr()),
        rewriter.getI64IntegerAttr(inputLevel),
        rewriter.getI64IntegerAttr(plan->bs),
        rewriter.getI64IntegerAttr(plan->gs),
        rewriter.getBoolAttr(plan->minKs));
    rewriter.replaceOp(op, result.getResult());
    return success();
  }

 private:
  bool enableMinKs;
};

struct ConvertKernelPrepareLinearTransformOp
    : public OpConversionPattern<kernel::PrepareLinearTransformOp> {
  ConvertKernelPrepareLinearTransformOp(const TypeConverter& converter,
                                        MLIRContext* context, bool enableMinKs)
      : OpConversionPattern(converter, context), enableMinKs(enableMinKs) {}

  LogicalResult matchAndRewrite(
      kernel::PrepareLinearTransformOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = getContextualContext(op.getOperation());
    if (failed(ctx)) return failure();
    auto preparedType = op.getPrepared().getType();
    double ratio = preparedType.getLogBsgsRatio() == 0
                       ? 4.0
                       : std::exp2(preparedType.getLogBsgsRatio());
    auto plan = getLinearTransformPlan(op, op.getDiagonals().getType(),
                                       op.getDiagonalIndices(), ratio,
                                       enableMinKs, rewriter);
    if (failed(plan)) return failure();

    Type resultType = typeConverter->convertType(preparedType);
    Value dest = makeEmptyDest(rewriter, op.getLoc(), resultType);
    auto result = cheddar::PrepareLinearTransformOp::create(
        rewriter, op.getLoc(), resultType, ctx.value(), adaptor.getDiagonals(),
        dest, plan->diagonalIndices,
        convertI64ArrayAttr(rewriter, op.getSourceRowIndicesAttr()),
        rewriter.getI64IntegerAttr(plan->width),
        rewriter.getI64IntegerAttr(preparedType.getLevel()),
        rewriter.getI64IntegerAttr(plan->bs),
        rewriter.getI64IntegerAttr(plan->gs),
        rewriter.getBoolAttr(plan->minKs));
    rewriter.replaceOp(op, result.getResult());
    return success();
  }

 private:
  bool enableMinKs;
};

struct ConvertKernelApplyLinearTransformOp
    : public OpConversionPattern<kernel::ApplyLinearTransformOp> {
  ConvertKernelApplyLinearTransformOp(const TypeConverter& converter,
                                      MLIRContext* context, bool enableMinKs)
      : OpConversionPattern(converter, context), enableMinKs(enableMinKs) {}

  LogicalResult matchAndRewrite(
      kernel::ApplyLinearTransformOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = getContextualContext(op.getOperation());
    if (failed(ctx)) return failure();
    auto evkMap = getContextualArg<cheddar::EvkMapType>(op.getOperation());
    if (failed(evkMap)) return failure();
    Type resultType = typeConverter->convertType(op.getOutput().getType());
    Value dest =
        makeReusableDest(rewriter, op.getLoc(), resultType, adaptor.getInput());
    auto preparedType = op.getPrepared().getType();
    auto diagonalIndices = op.getDiagonalIndicesAttr();
    auto diagonalWidth = op.getDiagonalWidthAttr();
    if (!diagonalIndices || !diagonalWidth)
      return op.emitOpError(
          "requires diagonal_indices and diagonal_width planning metadata");
    double ratio = preparedType.getLogBsgsRatio() == 0
                       ? 4.0
                       : std::exp2(preparedType.getLogBsgsRatio());
    auto plan = getLinearTransformPlan(op, diagonalWidth.getInt(),
                                       diagonalIndices.asArrayRef(), ratio,
                                       enableMinKs, rewriter);
    if (failed(plan)) return failure();
    auto result = cheddar::ApplyPreparedLinearTransformOp::create(
        rewriter, op.getLoc(), resultType, ctx.value(), adaptor.getInput(),
        evkMap.value(), adaptor.getPrepared(), dest,
        rewriter.getBoolAttr(plan->minKs));
    rewriter.replaceOp(op, result.getResult());
    return success();
  }

 private:
  bool enableMinKs;
};

// kernel.eval_chebyshev -> cheddar.eval_poly. Both operations use Chebyshev
// coefficients on [-1, 1].
struct ConvertKernelEvalChebyshevOp
    : public OpConversionPattern<kernel::EvalChebyshevOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      kernel::EvalChebyshevOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = getContextualContext(op.getOperation());
    if (failed(ctx)) return ctx;
    auto evkMap = getContextualArg<cheddar::EvkMapType>(op.getOperation());
    if (failed(evkMap)) return evkMap;

    // The kernel op's upstream level interface is the single source of truth
    // for Chebyshev depth. CKKS level analysis applies the same value to the
    // ciphertext result type.
    int64_t requiredLevels = op.getLevelsToDrop();
    if (requiredLevels < 2)
      return op.emitOpError(
          "scale-snu CHEDDAR EvalPoly requires an effective degree of at "
          "least two");
    Type resultType = typeConverter->convertType(op.getOutput().getType());
    Value dest =
        makeReusableDest(rewriter, op.getLoc(), resultType, adaptor.getInput());
    auto result = cheddar::EvalPolyOp::create(
        rewriter, op.getLoc(), resultType, ctx.value(), adaptor.getInput(),
        evkMap.value(), dest, op.getCoefficientsAttr(),
        rewriter.getI64IntegerAttr(requiredLevels));
    rewriter.replaceOp(op, result.getResult());
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
    auto result = tensor::ExtractSliceOp::create(rewriter, op.getLoc(), resTy,
                                                 adaptor.getTensor(), offsets,
                                                 sizes, strides);
    rewriter.replaceOp(op, result);
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
    auto result = tensor::InsertSliceOp::create(
        rewriter, op.getLoc(), adaptor.getScalar(), adaptor.getDest(), offsets,
        sizes, strides);
    rewriter.replaceOp(op, result);
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

// A bare LWE payload converts to a rank-0 tensor in Cheddar's DPS form.  When
// tensor.splat packs such a payload, extract the Cheddar scalar first; the
// generic structural conversion would otherwise feed tensor<!cheddar.X> to an
// op whose operand must be !cheddar.X.
struct ConvertPayloadSplat : public OpConversionPattern<tensor::SplatOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      tensor::SplatOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto inputTy = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
    auto resultTy =
        dyn_cast<RankedTensorType>(typeConverter->convertType(op.getType()));
    if (!inputTy || inputTy.getRank() != 0 ||
        !isCheddarPayload(inputTy.getElementType()) || !resultTy ||
        !isCheddarPayload(resultTy.getElementType()))
      return failure();

    auto scalar = tensor::ExtractOp::create(rewriter, op.getLoc(),
                                            adaptor.getInput(), ValueRange{});
    auto splat = tensor::SplatOp::create(rewriter, op.getLoc(), resultTy,
                                         scalar, ValueRange{});
    rewriter.replaceOp(op, splat.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Context-argument threading (mirrors LWEToLattigo)
//===----------------------------------------------------------------------===//

struct AddCheddarContextArg : public OpConversionPattern<func::FuncOp> {
  AddCheddarContextArg(
      const TypeConverter& typeConverter, mlir::MLIRContext* context,
      const std::vector<std::pair<Type, OpPredicate>>& evaluators)
      : OpConversionPattern<func::FuncOp>(typeConverter, context,
                                          /* benefit= */ 2),
        evaluators(evaluators) {}
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::FuncOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    SmallVector<Type, 4> selectedTypes;
    for (const auto& evaluator : evaluators) {
      if (!evaluator.second(op)) continue;
      // Skip a context type already present as an argument: with --debug
      // enabled the lwe debug port adds an LWESecretKey arg that converts to a
      // UserInterface, which would otherwise be duplicated here (and break
      // call- site threading, which dedups by type).
      if (llvm::any_of(op.getArgumentTypes(), [&](Type type) {
            return getTypeConverter()->convertType(type) == evaluator.first;
          }))
        continue;
      selectedTypes.push_back(evaluator.first);
    }
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
      const TypeConverter& typeConverter, mlir::MLIRContext* context,
      const std::vector<std::pair<Type, OpPredicate>>& evaluators)
      : OpConversionPattern<func::CallOp>(typeConverter, context),
        evaluators(evaluators) {}

  LogicalResult matchAndRewrite(
      func::CallOp op, typename func::CallOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto funcOp = getCalledFunction(op);
    if (failed(funcOp))
      return rewriter.notifyMatchFailure(op, "could not find callee function");
    SmallVector<Value> newOperands;
    for (const auto& evaluator : evaluators) {
      auto result =
          getContextualArgFromFunc(op.getOperation(), evaluator.first);
      if (failed(result)) continue;
      if (!llvm::is_contained(funcOp.value().getArgumentTypes(),
                              evaluator.first))
        continue;
      newOperands.push_back(result.value());
    }
    llvm::append_range(newOperands, adaptor.getOperands());
    // The callee's result types are type-converted by the structural func
    // pattern (e.g. tensor<Nx!lwe.plaintext> -> tensor<Nx!cheddar.plaintext>),
    // so the rebuilt call must use the converted result types or it stays
    // signature-inconsistent with its callee and fails to legalize. Operand
    // types come pre-converted via the adaptor.
    SmallVector<Type> newResultTypes;
    if (failed(
            typeConverter->convertTypes(op.getResultTypes(), newResultTypes)))
      return rewriter.notifyMatchFailure(op, "failed to convert result types");
    SmallVector<NamedAttribute> dialectAttrs(op->getDialectAttrs());
    auto call = func::CallOp::create(rewriter, op.getLoc(), op.getCallee(),
                                     newResultTypes, newOperands);
    call->setDialectAttrs(dialectAttrs);
    rewriter.replaceOp(op, call);
    return success();
  }

 private:
  std::vector<std::pair<Type, OpPredicate>> evaluators;
};

//===----------------------------------------------------------------------===//
// Debug port (__heir_debug_*) handling
//===----------------------------------------------------------------------===//
//
// `lwe-add-debug-port` lowers each `debug.validate` to a call to an external
// `func.func private @__heir_debug_N(%key: !lwe.secret_key, %ct:
// !lwe.ciphertext)`. To decrypt AND decode the ciphertext for printing, the
// CHEDDAR-side hook needs an `Encoder` (decode) and a `UserInterface`
// (decrypt), so we re-shape both the external declaration and every call site
// to
// `(Encoder, UserInterface, Ciphertext)`. The original `!lwe.secret_key`
// operand (which converts to a UserInterface) is dropped in favour of the
// contextual UserInterface threaded into the enclosing function, matching how
// all other CHEDDAR ops obtain their context args. The CheddarToEmitC pass then
// emits these calls as `__heir_debug(...)` C++ calls.

// The external __heir_debug_* declaration: reshape its signature to
// (Encoder, UserInterface, <converted ciphertext>) -> (). The ciphertext is the
// original last operand (a scalar `!lwe.ciphertext` -> rank-0
// `tensor<!cheddar.ciphertext>`, or a packed `tensor<Nx!lwe.ciphertext>` ->
// `tensor<Nx!cheddar.ciphertext>`), converted via the type converter.
struct ConvertDebugFuncDecl : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      func::FuncOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    if (!op.isExternal() || !isDebugPort(op.getName())) return failure();
    auto inputs = op.getFunctionType().getInputs();
    if (inputs.empty()) return failure();
    Type ctType = typeConverter->convertType(inputs.back());
    if (!ctType) return failure();
    auto* ctx = getContext();
    SmallVector<Type> argTypes{cheddar::EncoderType::get(ctx),
                               cheddar::UserInterfaceType::get(ctx), ctType};
    rewriter.modifyOpInPlace(op, [&] {
      op.setType(FunctionType::get(ctx, argTypes, {}));
      // __heir_debug only READS the ciphertext (decrypt+decode for printing).
      // Without this, one-shot-bufferize treats the external call's operand
      // conservatively as possibly-written and materializes a copy -- which for
      // a move-only cheddar Ciphertext becomes a destructive std::move, leaving
      // the observed value (and its later uses) empty -> "num primes mismatch".
      // Mark the ciphertext arg read-only so bufferization borrows it.
      op.setArgAttr(2, "bufferization.access", rewriter.getStringAttr("read"));
    });
    return success();
  }
};

// A call to __heir_debug_*: thread the enclosing function's Encoder +
// UserInterface contextual args, followed by the (converted) ciphertext operand
// (the last original operand; the original secret-key operand is dropped).
struct ConvertDebugCall : public OpConversionPattern<func::CallOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      func::CallOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    if (!isDebugPort(op.getCallee())) return failure();
    auto* ctx = getContext();
    auto encoder = getContextualArgFromFunc(op.getOperation(),
                                            cheddar::EncoderType::get(ctx));
    if (failed(encoder)) return failure();
    auto ui = getContextualArgFromFunc(op.getOperation(),
                                       cheddar::UserInterfaceType::get(ctx));
    if (failed(ui)) return failure();
    SmallVector<Value> newOperands{encoder.value(), ui.value(),
                                   adaptor.getOperands().back()};
    SmallVector<NamedAttribute> dialectAttrs(op->getDialectAttrs());
    auto call = func::CallOp::create(rewriter, op.getLoc(), op.getCallee(),
                                     TypeRange{}, newOperands);
    call->setDialectAttrs(dialectAttrs);
    rewriter.replaceOp(op, call);
    return success();
  }
};

}  // namespace

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

#define GEN_PASS_DEF_LWETOCHEDDAR
#include "lib/Dialect/LWE/Conversions/LWEToCheddar/LWEToCheddar.h.inc"

namespace {

struct LWEToCheddar : public impl::LWEToCheddarBase<LWEToCheddar> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();
    ToCheddarTypeConverter typeConverter(context);

    if (!moduleIsCKKS(module)) {
      module->emitOpError("CHEDDAR backend only supports CKKS scheme");
      return signalPassFailure();
    }

    // Configure bootstrap transforms for the slots actually represented by
    // each ciphertext, not the enclosing RLWE ring's polynomial degree.
    // Preparing twice the CKKS slot capacity wastes FFT transforms and rotation
    // keys, and some backends reject it as exceeding the ring maximum.
    std::optional<int64_t> bootstrapSlots;
    WalkResult slotWalk =
        module->walk([&](ckks::BootstrapOp op) {
          auto ctType = dyn_cast<lwe::LWECiphertextType>(
              getElementTypeOrSelf(op.getInput().getType()));
          if (!ctType) return WalkResult::advance();
          // CKKS packs N / 2 complex slots in an RLWE ring with polynomial
          // modulus x^N + 1. SecretToCKKS currently uses that RLWE polynomial
          // for the plaintext-space type as well, so its degree is the ring
          // dimension, not the represented slot count.
          int64_t ringDegree = ctType.getPlaintextSpace()
                                   .getRing()
                                   .getPolynomialModulus()
                                   .getPolynomial()
                                   .getDegree();
          int64_t slots = ringDegree / 2;
          if (bootstrapSlots && *bootstrapSlots != slots) {
            op.emitOpError()
                << "mixed bootstrap slot counts are not yet supported: "
                << *bootstrapSlots << " and " << slots;
            return WalkResult::interrupt();
          }
          bootstrapSlots = slots;
          return WalkResult::advance();
        });
    if (slotWalk.wasInterrupted()) return signalPassFailure();
    // scale-snu/cheddar requires at least 256 slots in its special-FFT
    // bootstrap implementation. Padding a smaller logical layout to that
    // backend minimum preserves its values while still avoiding the much
    // larger full-ring transform/key setup.
    constexpr int64_t kMinBootstrapSlots = 256;
    if (bootstrapSlots)
      module->setAttr(
          kActualSlotCountAttrName,
          IntegerAttr::get(IntegerType::get(context, 64),
                           std::max(*bootstrapSlots, kMinBootstrapSlots)));

    ConversionTarget target(*context);
    target.addLegalDialect<cheddar::CheddarDialect>();
    target.addLegalDialect<bufferization::BufferizationDialect>();
    target.addIllegalDialect<ckks::CKKSDialect, lwe::LWEDialect>();
    target.addIllegalOp<
        kernel::LinearTransformOp, kernel::PrepareLinearTransformOp,
        kernel::ApplyLinearTransformOp, kernel::EvalChebyshevOp>();
    // preprocessing.* ops are legal once their plaintext element types have
    // been converted to cheddar's; --preprocessing-to-cheddar lowers them
    // after.
    target.addDynamicallyLegalDialect<preprocessing::PreprocessingDialect>(
        [&](Operation* op) { return typeConverter.isLegal(op); });
    RewritePatternSet patterns(context);
    addStructuralConversionPatterns(typeConverter, patterns, target);
    addTensorConversionPatterns(typeConverter, patterns, target);
    preprocessing::populatePreprocessingConversions(patterns, typeConverter,
                                                    context);

    // BootContext and the rotation-key map (EvkMap) must be threaded
    // transitively: a function that (directly or indirectly) calls a function
    // that bootstraps / needs the key map must itself carry those context
    // arguments so it can forward them. A per-body walk only sees a function's
    // own ops (the bootstrap may live in a callee, e.g. @main calling
    // @main__preprocessed), so propagate both properties to all transitive
    // callers via a fixed point over the call graph.
    DenseMap<func::FuncOp, bool> bootstrapsTransitively;
    DenseMap<func::FuncOp, bool> needsEvkMapTransitively;
    module->walk([&](func::FuncOp f) {
      bool boots = false;
      bool evk = false;
      f.walk([&](Operation* inner) {
        if (isa<ckks::BootstrapOp>(inner)) boots = true;
        if (isa<ckks::BootstrapOp, kernel::LinearTransformOp,
                kernel::ApplyLinearTransformOp, kernel::EvalChebyshevOp>(inner))
          evk = true;
      });
      bootstrapsTransitively[f] = boots;
      needsEvkMapTransitively[f] = evk;
    });
    bool changed = true;
    while (changed) {
      changed = false;
      module->walk([&](func::CallOp call) {
        auto caller = call->getParentOfType<func::FuncOp>();
        FailureOr<func::FuncOp> callee = getCalledFunction(call);
        if (!caller || failed(callee)) return;
        if (bootstrapsTransitively[callee.value()] &&
            !bootstrapsTransitively[caller]) {
          bootstrapsTransitively[caller] = true;
          changed = true;
        }
        if (needsEvkMapTransitively[callee.value()] &&
            !needsEvkMapTransitively[caller]) {
          needsEvkMapTransitively[caller] = true;
          changed = true;
        }
      });
    }

    auto hasCryptoOps = [&](Operation* op) -> bool {
      return containsArgumentOfDialect<lwe::LWEDialect, ckks::CKKSDialect>(op);
    };
    auto hasPrepareLinearTransform = [&](Operation* op) -> bool {
      auto funcOp = dyn_cast<func::FuncOp>(op);
      if (!funcOp) return false;
      return !funcOp.getOps<kernel::PrepareLinearTransformOp>().empty();
    };
    auto hasEncodeOps = [&](Operation* op) -> bool {
      auto funcOp = dyn_cast<func::FuncOp>(op);
      if (!funcOp) return false;
      bool found = false;
      funcOp->walk([&](lwe::RLWEEncodeOp) { found = true; });
      return found;
    };
    auto needsEvkMap = [&needsEvkMapTransitively](Operation* op) -> bool {
      auto funcOp = dyn_cast<func::FuncOp>(op);
      return funcOp && needsEvkMapTransitively.lookup(funcOp);
    };
    auto funcBootstraps = [&bootstrapsTransitively](Operation* op) -> bool {
      auto funcOp = dyn_cast<func::FuncOp>(op);
      return funcOp && bootstrapsTransitively.lookup(funcOp);
    };
    auto hasCryptoOrEncode = [&](Operation* op) {
      return hasCryptoOps(op) || hasEncodeOps(op);
    };
    auto hasContextOps = [&](Operation* op) {
      return hasCryptoOps(op) || hasPrepareLinearTransform(op);
    };
    std::vector<std::pair<Type, OpPredicate>> evaluators = {
        {cheddar::ContextType::get(context), hasContextOps},
        {cheddar::BootContextType::get(context), funcBootstraps},
        {cheddar::EncoderType::get(context), hasCryptoOrEncode},
        {cheddar::UserInterfaceType::get(context), hasCryptoOps},
        {cheddar::EvalKeyType::get(context), hasCryptoOps},
        {cheddar::EvkMapType::get(context), needsEvkMap},
    };

    patterns.add<AddCheddarContextArg>(typeConverter, context, evaluators);
    patterns.add<ConvertCheddarFuncCallOp>(typeConverter, context, evaluators);
    // Debug ports get dedicated, higher-benefit handling (the generic call /
    // structural func patterns would mis-thread their context args).
    patterns.add<ConvertDebugFuncDecl, ConvertDebugCall>(typeConverter, context,
                                                         /*benefit=*/3);

    patterns.add<ConvertCKKSAddOp, ConvertCKKSSubOp, ConvertCKKSMulOp,
                 ConvertCKKSAddPlainOp, ConvertCKKSSubPlainOp,
                 ConvertCKKSMulPlainOp, ConvertCKKSNegateOp, ConvertCKKSRelinOp,
                 ConvertCKKSRescaleOp, ConvertCKKSRotateOp,
                 ConvertCKKSLevelReduceOp, ConvertCKKSBootstrapOp>(
        typeConverter, context);
    patterns.add<ConvertRAddOp, ConvertRSubOp, ConvertRMulOp, ConvertRNegateOp,
                 ConvertRAddPlainOp, ConvertRSubPlainOp, ConvertRMulPlainOp>(
        typeConverter, context);
    patterns.add<ConvertLWEEncodeOp, ConvertLWEDecodeOp, ConvertLWEEncryptOp,
                 ConvertLWEDecryptOp>(typeConverter, context);
    patterns.add<ConvertKernelLinearTransformOp,
                 ConvertKernelPrepareLinearTransformOp,
                 ConvertKernelApplyLinearTransformOp>(typeConverter, context,
                                                      enableMinKs);
    patterns.add<ConvertKernelEvalChebyshevOp>(typeConverter, context);
    // Payload packing ops -> rank-reducing slice ops (benefit 2 so they win
    // over the structural tensor conversion for payload-typed tensors).
    patterns.add<ConvertPayloadExtract, ConvertPayloadInsert,
                 ConvertPayloadFromElements, ConvertPayloadSplat>(
        typeConverter, context, /*benefit=*/2);

    // A reshaped `__heir_debug_*` declaration / call has exactly
    // (Encoder, UserInterface, tensor<...x!cheddar.ciphertext>) inputs and no
    // results.
    auto isReshapedDebugSig = [](TypeRange ins) {
      if (ins.size() != 3 || !isa<cheddar::EncoderType>(ins[0]) ||
          !isa<cheddar::UserInterfaceType>(ins[1]))
        return false;
      auto t = dyn_cast<RankedTensorType>(ins[2]);
      return t && isa<cheddar::CiphertextType>(t.getElementType());
    };
    auto isReshapedDebugDecl = [&](func::FuncOp op) {
      return op.getFunctionType().getNumResults() == 0 &&
             isReshapedDebugSig(op.getFunctionType().getInputs());
    };
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      if (isDebugPort(op.getName())) return isReshapedDebugDecl(op);
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
      bool hasPrepareOp =
          !op.getOps<kernel::PrepareLinearTransformOp>().empty();
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody()) &&
             (!(hasCryptoArg || hasEncodeOp || hasPrepareOp) ||
              hasCheddarCtxArg);
    });

    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      if (isDebugPort(op.getCallee()))
        return isReshapedDebugSig(op.getCalleeType().getInputs());
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
