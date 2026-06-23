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
#include "llvm/include/llvm/ADT/APInt.h"                 // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
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
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "lwe-to-cheddar"

namespace mlir::heir::lwe {

//===----------------------------------------------------------------------===//
// Type converter
//===----------------------------------------------------------------------===//

class ToCheddarTypeConverter : public TypeConverter {
 public:
  ToCheddarTypeConverter(MLIRContext* ctx) {
    addConversion([](Type type) { return type; });
    addConversion([ctx](lwe::LWECiphertextType type) -> Type {
      return cheddar::CiphertextType::get(ctx);
    });
    addConversion([ctx](lwe::LWEPlaintextType type) -> Type {
      return cheddar::PlaintextType::get(ctx);
    });
    // Keys are handled differently in CHEDDAR (EvkMap + individual EvalKeys),
    // but during conversion we keep them as-is and let AddEvaluatorArg handle
    // the key threading.
    addConversion([ctx](lwe::LWEPublicKeyType type) -> Type {
      // Public key is not directly used — absorbed into UserInterface
      return cheddar::UserInterfaceType::get(ctx);
    });
    addConversion([ctx](lwe::LWESecretKeyType type) -> Type {
      // Secret key is not directly used — absorbed into UserInterface
      return cheddar::UserInterfaceType::get(ctx);
    });
    addConversion([this](RankedTensorType type) -> Type {
      return RankedTensorType::get(type.getShape(),
                                   this->convertType(type.getElementType()));
    });
  }
};

//===----------------------------------------------------------------------===//
// Helper: get contextual arguments by type
//===----------------------------------------------------------------------===//

namespace {

template <typename... Dialects>
bool containsArgumentOfDialect(Operation* op) {
  auto funcOp = dyn_cast<func::FuncOp>(op);
  if (!funcOp) {
    return false;
  }
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

// The server-side context argument, accepting either a plain `!cheddar.context`
// or a `!cheddar.boot_context`. A bootstrapping function carries a BootContext
// (which all the ordinary ops in it still run on); a non-boot function carries
// a plain Context. cheddar.boot itself separately requires the BootContext.
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

    rewriter.replaceOpWithNewOp<CheddarOp>(
        op, this->typeConverter->convertType(op.getOutput().getType()),
        ctx.value(), adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

using ConvertCKKSAddOp = ConvertCKKSBinOp<ckks::AddOp, cheddar::AddOp>;
using ConvertCKKSSubOp = ConvertCKKSBinOp<ckks::SubOp, cheddar::SubOp>;
using ConvertCKKSMulOp = ConvertCKKSBinOp<ckks::MulOp, cheddar::MultOp>;

// LWE R* ops (used by torch-linalg-to-ckks pipeline)
using ConvertRAddOp = ConvertCKKSBinOp<lwe::RAddOp, cheddar::AddOp>;
using ConvertRSubOp = ConvertCKKSBinOp<lwe::RSubOp, cheddar::SubOp>;
using ConvertRMulOp = ConvertCKKSBinOp<lwe::RMulOp, cheddar::MultOp>;

// Ct-pt operations
template <typename CKKSOp, typename CheddarOp>
struct ConvertCKKSPlainOp : public OpConversionPattern<CKKSOp> {
  using OpConversionPattern<CKKSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      CKKSOp op, typename CKKSOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = getContextualContext(op.getOperation());
    if (failed(ctx)) return ctx;

    // Ensure ciphertext is first operand (CHEDDAR convention)
    Value ciphertext = adaptor.getLhs();
    Value plaintext = adaptor.getRhs();
    if (!isa<cheddar::CiphertextType>(adaptor.getLhs().getType())) {
      ciphertext = adaptor.getRhs();
      plaintext = adaptor.getLhs();
    }

    rewriter.replaceOpWithNewOp<CheddarOp>(
        op, this->typeConverter->convertType(op.getOutput().getType()),
        ctx.value(), ciphertext, plaintext);
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

// Negate
struct ConvertCKKSNegateOp : public OpConversionPattern<ckks::NegateOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ckks::NegateOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = getContextualContext(op.getOperation());
    if (failed(ctx)) return ctx;

    rewriter.replaceOpWithNewOp<cheddar::NegOp>(
        op, this->typeConverter->convertType(op.getOutput().getType()),
        ctx.value(), adaptor.getInput());
    return success();
  }
};

// Relinearize — needs the multiplication key.
//
// The CHEDDAR multiplication (relinearization) key is threaded through the
// enclosing function as a `!cheddar.eval_key` argument (see the evaluator list
// in the pass below), exactly like the hand-written CHEDDAR kernels take a
// `%mult_key` parameter. We look it up by type rather than materialising it
// from the UserInterface via `cheddar.get_mult_key`: the cheddar-to-emitc
// lowering rejects `get_mult_key` because it returns a const reference to a
// move-only value that cannot be bound to a local.
struct ConvertCKKSRelinOp : public OpConversionPattern<ckks::RelinearizeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ckks::RelinearizeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = getContextualContext(op.getOperation());
    if (failed(ctx)) return ctx;
    auto multKey = getContextualArg<cheddar::EvalKeyType>(op.getOperation());
    if (failed(multKey)) return multKey;

    rewriter.replaceOpWithNewOp<cheddar::RelinearizeOp>(
        op, this->typeConverter->convertType(op.getOutput().getType()),
        ctx.value(), adaptor.getInput(), multKey.value());
    return success();
  }
};

// Rescale (mod reduce in CKKS)
struct ConvertCKKSRescaleOp : public OpConversionPattern<ckks::RescaleOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ckks::RescaleOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = getContextualContext(op.getOperation());
    if (failed(ctx)) return ctx;

    rewriter.replaceOpWithNewOp<cheddar::RescaleOp>(
        op, this->typeConverter->convertType(op.getOutput().getType()),
        ctx.value(), adaptor.getInput());
    return success();
  }
};

// Rotate
//
// The dialect no longer carries a rotation-key SSA operand: the
// cheddar-to-emitc lowering looks up `ui->GetRotationKey(distance)` inline at
// the use site. We just build a keyless `cheddar.hrot` carrying the static or
// dynamic distance.
struct ConvertCKKSRotateOp : public OpConversionPattern<ckks::RotateOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ckks::RotateOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = getContextualContext(op.getOperation());
    if (failed(ctx)) return ctx;

    Value dynamicShift = adaptor.getDynamicShift();
    IntegerAttr staticShift = op.getStaticShiftAttr();
    if (!staticShift && !dynamicShift) {
      return rewriter.notifyMatchFailure(
          op, "rotate op must have either static or dynamic shift");
    }

    if (dynamicShift) {
      rewriter.replaceOpWithNewOp<cheddar::HRotOp>(
          op, this->typeConverter->convertType(op.getOutput().getType()),
          ctx.value(), adaptor.getInput(), dynamicShift,
          /*static_distance=*/IntegerAttr());
    } else {
      rewriter.replaceOpWithNewOp<cheddar::HRotOp>(
          op, this->typeConverter->convertType(op.getOutput().getType()),
          ctx.value(), adaptor.getInput(), /*dynamic_distance=*/Value(),
          staticShift);
    }
    return success();
  }
};

// Level reduce
struct ConvertCKKSLevelReduceOp
    : public OpConversionPattern<ckks::LevelReduceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ckks::LevelReduceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = getContextualContext(op.getOperation());
    if (failed(ctx)) return ctx;

    // Derive target level from the output ciphertext's modulus chain.
    auto outputCtType = dyn_cast<lwe::LWECiphertextType>(
        getElementTypeOrSelf(op.getOutput().getType()));
    int64_t targetLevelVal =
        outputCtType ? outputCtType.getModulusChain().getCurrent() : 0;
    auto targetLevel = rewriter.getI64IntegerAttr(targetLevelVal);
    rewriter.replaceOpWithNewOp<cheddar::LevelDownOp>(
        op, this->typeConverter->convertType(op.getOutput().getType()),
        ctx.value(), adaptor.getInput(), targetLevel);
    return success();
  }
};

// Bootstrap
//
// NOTE: This creates a `cheddar.get_evk_map` op to obtain the EvkMap from the
// UserInterface. The cheddar-to-emitc lowering currently rejects
// `get_evk_map` (same const-reference-to-move-only issue as get_mult_key), so
// kernels that actually bootstrap will not lower to C++ yet. Bootstrapping is
// not on the MNIST MLP critical path; if it becomes needed, thread the EvkMap
// as a contextual function argument the way the mult key is handled above.
struct ConvertCKKSBootstrapOp : public OpConversionPattern<ckks::BootstrapOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ckks::BootstrapOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = getContextualContext(op.getOperation());
    if (failed(ctx)) return ctx;
    // The EvkMap (bootstrapping keys) is threaded as a contextual
    // `!cheddar.evk_map` function argument -- same as the mult key -- rather
    // than materialised via a `cheddar.get_evk_map` getter (which the emitter
    // rejects, being a const ref to a move-only value).
    auto evkMap = getContextualArg<cheddar::EvkMapType>(op.getOperation());
    if (failed(evkMap)) return evkMap;

    rewriter.replaceOpWithNewOp<cheddar::BootOp>(
        op, this->typeConverter->convertType(op.getOutput().getType()),
        ctx.value(), adaptor.getInput(), evkMap.value());
    return success();
  }
};

// Encode. CHEDDAR's Encode(pt, level, scale, msg) needs the plaintext level and
// its scale. This pattern is a pure forwarder: it does NOT compute or adjust
// scales, it only forwards the exact scale and per-use level that the upstream
// CKKS scale management already put on the plaintext type / op. CHEDDAR asserts
// the scale matches its own canonical value (rejecting mismatches beyond
// 1e-12); whether that value is correct is the higher-level pipeline's job, so
// fixing the upstream scale management never requires changing this lowering.
struct ConvertLWEEncodeOp : public OpConversionPattern<lwe::RLWEEncodeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      lwe::RLWEEncodeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto encoder = getContextualArg<cheddar::EncoderType>(op.getOperation());
    if (failed(encoder)) return encoder;

    // We still require an InverseCanonicalEncoding to confirm this is a CKKS
    // plaintext; its (nominal) scaling factor is forwarded into the op's
    // vestigial scale attr, but the emitter ignores it and encodes at the
    // canonical per-level scale ctx->param_.GetScale(level) instead.
    auto ptType = dyn_cast<lwe::LWEPlaintextType>(op.getOutput().getType());
    auto invEncoding = ptType ? dyn_cast<lwe::InverseCanonicalEncodingAttr>(
                                    ptType.getPlaintextSpace().getEncoding())
                              : nullptr;
    if (!invEncoding)
      return op.emitOpError()
             << "cannot lower to cheddar.encode: plaintext has no "
                "InverseCanonicalEncoding (not a CKKS plaintext)";
    double scale = static_cast<double>(invEncoding.getScalingFactor());

    // Per-use level: the RLWEEncodeOp level attr is set from the consuming
    // ciphertext's modulus chain (Patterns.cpp / ContextAwareConversionUtils),
    // i.e. the level the standard level analysis assigned. The cheddar emitter
    // uses it to encode at the canonical per-level scale GetScale(level).
    int64_t level = op.getLevel() ? op.getLevel().value() : 0;

    auto ptTy = cheddar::PlaintextType::get(getContext());
    rewriter.replaceOpWithNewOp<cheddar::EncodeOp>(
        op, ptTy, encoder.value(), adaptor.getInput(),
        rewriter.getI64IntegerAttr(level), rewriter.getF64FloatAttr(scale));
    return success();
  }
};

// Decrypt
struct ConvertLWEDecryptOp : public OpConversionPattern<lwe::RLWEDecryptOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      lwe::RLWEDecryptOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ui = getContextualArg<cheddar::UserInterfaceType>(op.getOperation());
    if (failed(ui)) return ui;

    rewriter.replaceOpWithNewOp<cheddar::DecryptOp>(
        op, cheddar::PlaintextType::get(getContext()), ui.value(),
        adaptor.getInput());
    return success();
  }
};

// Encrypt
struct ConvertLWEEncryptOp : public OpConversionPattern<lwe::RLWEEncryptOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      lwe::RLWEEncryptOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ui = getContextualArg<cheddar::UserInterfaceType>(op.getOperation());
    if (failed(ui)) return ui;

    rewriter.replaceOpWithNewOp<cheddar::EncryptOp>(
        op, cheddar::CiphertextType::get(getContext()), ui.value(),
        adaptor.getInput());
    return success();
  }
};

// Decode
struct ConvertLWEDecodeOp : public OpConversionPattern<lwe::RLWEDecodeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      lwe::RLWEDecodeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto encoder = getContextualArg<cheddar::EncoderType>(op.getOperation());
    if (failed(encoder)) return encoder;

    // cheddar.decode is destination-passing: provide an empty destination
    // buffer ($value) that the decoded result aliases, so the message path
    // bufferizes cleanly (see Cheddar Transforms/BufferizableOpInterfaceImpl).
    auto outTy = cast<RankedTensorType>(op.getOutput().getType());
    Value dest = tensor::EmptyOp::create(
        rewriter, op.getLoc(), outTy.getShape(), outTy.getElementType());
    rewriter.replaceOpWithNewOp<cheddar::DecodeOp>(op, outTy, encoder.value(),
                                                   adaptor.getInput(), dest);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Orion extension ops (linear_transform, chebyshev) -> CHEDDAR extension ops
//===----------------------------------------------------------------------===//

// orion.linear_transform -> cheddar.linear_transform (CHEDDAR's BSGS+hoisting
// LinearTransform extension). The diagonals stay a cleartext operand; the
// EvkMap (rotation keys) is threaded as a contextual function argument.
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

    // CHEDDAR's LinearTransform operates at the input level and does NOT
    // rescale internally (the explicit cheddar.rescale that follows is the only
    // level drop), so the op level is just the Orion layer level.
    auto level = rewriter.getI64IntegerAttr(op.getOrionLevelAttr().getInt());

    // Pick a principled BSGS grid: a square-ish split of the rotation span,
    // biased by the requested baby:giant ratio. gs is the giant-step count, bs
    // the baby-step count, and bs*gs covers the largest diagonal index. This is
    // a sensible default for a dense transform; a sparse/wrap-around transform
    // may be hand-tuned to gs=1 (pure diagonal method) in the produced IR. The
    // consumers (emitter, getRotationIndices) read bs/gs as-is and never
    // recompute them.
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
    int64_t bs = (need + gs - 1) / gs;  // ceil(need/gs) so bs*gs >= need

    rewriter.replaceOpWithNewOp<cheddar::LinearTransformOp>(
        op, this->typeConverter->convertType(op.getResult().getType()),
        ctx.value(), adaptor.getInput(), evkMap.value(), adaptor.getDiagonals(),
        op.getDiagonalIndicesAttr(), level, rewriter.getI64IntegerAttr(bs),
        rewriter.getI64IntegerAttr(gs));
    return success();
  }
};

// orion.chebyshev -> cheddar.eval_poly (CHEDDAR's EvalPoly extension).
//
// CHEDDAR's EvalPoly has no notion of an approximation interval: it evaluates a
// Chebyshev series on the canonical domain [-1, 1] and assumes the input
// ciphertext already lives there. The orion op, mirroring Lattigo, carries the
// domain [domain_start, domain_end] its coefficients were fit on and expects
// the evaluator to remap x in [a, b] to [-1, 1] (Lattigo does this internally;
// CHEDDAR cannot). Orion's frontend normalizes activations to [-1, 1] and emits
// that domain in practice. If a non-canonical domain ever reaches here we
// refuse to lower rather than silently drop it (which would miscompute the
// activation): with backend=cheddar the higher-level pipeline must materialize
// the affine input remap (mult_const/rescale/add_const to [-1, 1]) before this
// conversion.
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
    if (domainStart != -1.0 || domainEnd != 1.0) {
      return op.emitError()
             << "cannot lower orion.chebyshev with approximation domain ["
             << domainStart << ", " << domainEnd
             << "] to the cheddar backend: CHEDDAR's EvalPoly only evaluates "
                "on "
                "[-1, 1]. The higher-level pipeline must normalize the domain "
                "(materialize the affine input remap to [-1, 1]) before "
                "lowering "
                "to cheddar.";
    }

    // Input level (where the ciphertext enters) and output level (where the
    // polynomial's multiplicative depth leaves it) both come from the modulus
    // chains the upstream level analysis assigned; CHEDDAR's EvalPoly rescales
    // the result to the canonical scale of the output level. A missing modulus
    // chain means the upstream level analysis did not run -- fail rather than
    // silently emit level 0 (the top of the chain), which would be wrong.
    auto inTy = dyn_cast<lwe::LWECiphertextType>(op.getInput().getType());
    auto resTy = dyn_cast<lwe::LWECiphertextType>(op.getResult().getType());
    if (!inTy || !inTy.getModulusChain() || !resTy || !resTy.getModulusChain())
      return op.emitOpError()
             << "cannot lower to cheddar.eval_poly: input/result ciphertext "
                "has no modulus chain to read the EvalPoly input/output level "
                "from (run the upstream CKKS level analysis first)";
    int64_t level = inTy.getModulusChain().getCurrent();
    int64_t outputLevel = resTy.getModulusChain().getCurrent();

    rewriter.replaceOpWithNewOp<cheddar::EvalPolyOp>(
        op, this->typeConverter->convertType(op.getResult().getType()),
        ctx.value(), adaptor.getInput(), evkMap.value(),
        op.getCoefficientsAttr(), rewriter.getI64IntegerAttr(level),
        rewriter.getI64IntegerAttr(outputLevel));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AddEvaluatorArg pattern (mirrors LWEToLattigo)
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
    for (const auto& evaluator : evaluators) {
      if (evaluator.second(op)) {
        selectedTypes.push_back(evaluator.first);
      }
    }

    if (selectedTypes.empty()) {
      return rewriter.notifyMatchFailure(op, "no CHEDDAR context needed");
    }

    SmallVector<unsigned> argIndices(selectedTypes.size(), 0);
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
    if (failed(funcOp)) {
      return rewriter.notifyMatchFailure(op, "could not find callee function");
    }

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

}  // anonymous namespace

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

#define GEN_PASS_DEF_LWETOCHEDDAR
#include "lib/Dialect/LWE/Conversions/LWEToCheddar/LWEToCheddar.h.inc"

struct LWEToCheddar : public impl::LWEToCheddarBase<LWEToCheddar> {
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();
    ToCheddarTypeConverter typeConverter(context);

    // Only run for CKKS modules (CHEDDAR is CKKS-only)
    if (!moduleIsCKKS(module)) {
      module->emitOpError("CHEDDAR backend only supports CKKS scheme");
      return signalPassFailure();
    }

    ConversionTarget target(*context);
    target.addLegalDialect<cheddar::CheddarDialect>();
    target.addIllegalDialect<ckks::CKKSDialect, orion::OrionDialect>();
    target
        .addIllegalOp<lwe::RLWEEncryptOp, lwe::RLWEDecryptOp, lwe::RLWEEncodeOp,
                      lwe::RLWEDecodeOp, lwe::RAddOp, lwe::RSubOp, lwe::RMulOp,
                      lwe::RAddPlainOp, lwe::RSubPlainOp, lwe::RMulPlainOp>();

    RewritePatternSet patterns(context);
    addStructuralConversionPatterns(typeConverter, patterns, target);
    addTensorConversionPatterns(typeConverter, patterns, target);

    // Predicate: function contains CKKS/LWE operands
    auto hasCryptoOps = [&](Operation* op) -> bool {
      return containsArgumentOfDialect<lwe::LWEDialect, ckks::CKKSDialect>(op);
    };

    // Predicate: function contains encode ops
    auto hasEncodeOps = [&](Operation* op) -> bool {
      auto funcOp = dyn_cast<func::FuncOp>(op);
      if (!funcOp) return false;
      bool found = false;
      funcOp->walk([&](lwe::RLWEEncodeOp) { found = true; });
      return found;
    };

    // Predicate: function contains an op that needs the EvkMap threaded -- a
    // bootstrap, or an Orion linear_transform / chebyshev (CHEDDAR's
    // LinearTransform / EvalPoly extensions take an EvkMap of rotation keys).
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

    // Predicate: function bootstraps (contains a ckks.bootstrap). Such a
    // function gets a `!cheddar.boot_context` argument instead of a plain
    // `!cheddar.context`, so `cheddar.boot` can call BootContext::Boot with no
    // downcast (the ordinary ops in it still run on the same context).
    auto funcBootstraps = [&](Operation* op) -> bool {
      auto funcOp = dyn_cast<func::FuncOp>(op);
      if (!funcOp) return false;
      bool found = false;
      funcOp->walk([&](ckks::BootstrapOp) { found = true; });
      return found;
    };

    // Crypto function that does not bootstrap -> plain Context.
    auto hasCryptoNoBoot = [&](Operation* op) -> bool {
      return hasCryptoOps(op) && !funcBootstraps(op);
    };

    // CHEDDAR context args to thread through functions. The multiplication
    // (relinearization) key is threaded as a `!cheddar.eval_key` argument, and
    // the bootstrapping key map as a `!cheddar.evk_map` argument (only into
    // functions that bootstrap); rotation/conjugation keys are looked up inline
    // from the UserInterface by the cheddar-to-emitc lowering and are NOT
    // threaded here. The Encoder is threaded into any function that does crypto
    // OR encoding (an encode-only function -- the preprocessing function --
    // needs it for `encoder.Encode(...)`). The Context is threaded only into
    // crypto functions: the emitter emits the encode scale verbatim from the
    // IR, so encode-only functions no longer need the Context.
    auto hasCryptoOrEncode = [&](Operation* op) {
      return hasCryptoOps(op) || hasEncodeOps(op);
    };
    // Exactly one of Context / BootContext fires per function (a bootstrapping
    // function gets the BootContext); the resulting argument order is the same.
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

    // CKKS ops
    patterns.add<ConvertCKKSAddOp, ConvertCKKSSubOp, ConvertCKKSMulOp,
                 ConvertCKKSAddPlainOp, ConvertCKKSSubPlainOp,
                 ConvertCKKSMulPlainOp, ConvertCKKSNegateOp, ConvertCKKSRelinOp,
                 ConvertCKKSRescaleOp, ConvertCKKSRotateOp,
                 ConvertCKKSLevelReduceOp, ConvertCKKSBootstrapOp>(
        typeConverter, context);

    // LWE R* ops (produced by torch-linalg-to-ckks pipeline)
    patterns.add<ConvertRAddOp, ConvertRSubOp, ConvertRMulOp,
                 ConvertRAddPlainOp, ConvertRSubPlainOp, ConvertRMulPlainOp>(
        typeConverter, context);

    // LWE encrypt/decrypt/encode/decode ops
    patterns.add<ConvertLWEEncodeOp, ConvertLWEDecodeOp, ConvertLWEEncryptOp,
                 ConvertLWEDecryptOp>(typeConverter, context);

    // Orion extension ops (linear_transform, chebyshev)
    patterns.add<ConvertOrionLinearTransformOp, ConvertOrionChebyshevOp>(
        typeConverter, context);

    // Dynamically legal: func ops that have been converted. A function needs
    // a CHEDDAR context argument if it has crypto-typed args OR contains encode
    // ops (the encode-only preprocessing function takes only an encoder).
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

    // A call is legal only when the callee signature it carries is consistent
    // with the (possibly already converted) callee function. Without the
    // consistency check, a call whose operands have been converted to cheddar
    // types is wrongly considered legal before its context args are threaded,
    // leaving a stranded unrealized_conversion_cast.
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
      if (succeeded(callee)) {
        signatureConsistent =
            callee.value().getFunctionType() == op.getCalleeType();
      }
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

}  // namespace mlir::heir::lwe
