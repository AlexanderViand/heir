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
    auto ctx = getContextualArg<cheddar::ContextType>(op.getOperation());
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
    auto ctx = getContextualArg<cheddar::ContextType>(op.getOperation());
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
    auto ctx = getContextualArg<cheddar::ContextType>(op.getOperation());
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
    auto ctx = getContextualArg<cheddar::ContextType>(op.getOperation());
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
    auto ctx = getContextualArg<cheddar::ContextType>(op.getOperation());
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
    auto ctx = getContextualArg<cheddar::ContextType>(op.getOperation());
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
    auto ctx = getContextualArg<cheddar::ContextType>(op.getOperation());
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
    auto ctx = getContextualArg<cheddar::ContextType>(op.getOperation());
    if (failed(ctx)) return ctx;
    auto ui = getContextualArg<cheddar::UserInterfaceType>(op.getOperation());
    if (failed(ui)) return ui;

    auto evkMap = cheddar::GetEvkMapOp::create(
        rewriter, op.getLoc(), cheddar::EvkMapType::get(getContext()),
        ui.value());

    rewriter.replaceOpWithNewOp<cheddar::BootOp>(
        op, this->typeConverter->convertType(op.getOutput().getType()),
        ctx.value(), adaptor.getInput(), evkMap);
    return success();
  }
};

// Encode. CHEDDAR's Encode(pt, level, scale, msg) needs the plaintext level and
// its exact scale. We forward the precise per-use level and the (nominal) scale
// from the scale-management analysis. The cheddar-to-emitc emitter emits this
// scale verbatim, so the cheddar-level IR must carry the EXACT canonical scale
// CHEDDAR asserts against (it rejects mismatches beyond 1e-12) -- produced by
// precise scale management upstream, or for now baked into the test IR.
struct ConvertLWEEncodeOp : public OpConversionPattern<lwe::RLWEEncodeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      lwe::RLWEEncodeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto encoder = getContextualArg<cheddar::EncoderType>(op.getOperation());
    if (failed(encoder)) return encoder;

    // Scaling factor (APInt). It is the nominal canonical scale
    // (≈2^logDefaultScale) or a doubled scale (≈2^2logScale, e.g. post-mult
    // before rescale). The emitter emits it verbatim; the exact (prime-derived)
    // canonical value must already be baked in upstream.
    APInt scaleAP(64, 1ULL << 45);  // default 2^45
    auto ptType = dyn_cast<lwe::LWEPlaintextType>(op.getOutput().getType());
    if (ptType) {
      if (auto invEncoding = dyn_cast<lwe::InverseCanonicalEncodingAttr>(
              ptType.getPlaintextSpace().getEncoding())) {
        scaleAP = invEncoding.getScalingFactor().getValue();
      }
    }
    int64_t logScale = scaleAP.isPowerOf2()
                           ? static_cast<int64_t>(scaleAP.exactLogBase2())
                           : static_cast<int64_t>(scaleAP.nearestLogBase2());

    // Precise per-use level from the scale-management analysis (the optional
    // RLWEEncodeOp level attr). Requires running torch-linalg-to-ckks with
    // ckks-scale-policy=precise.
    int64_t level = op.getLevel() ? op.getLevel().value() : 0;

    auto ptTy = cheddar::PlaintextType::get(getContext());
    rewriter.replaceOpWithNewOp<cheddar::EncodeOp>(
        op, ptTy, encoder.value(), adaptor.getInput(),
        rewriter.getI64IntegerAttr(level),
        rewriter.getF64FloatAttr(std::pow(2.0, (double)logScale)));
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
    target.addIllegalDialect<ckks::CKKSDialect>();
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

    // CHEDDAR context args to thread through functions. The multiplication
    // (relinearization) key is threaded as a `!cheddar.eval_key` argument;
    // rotation/conjugation keys are looked up inline from the UserInterface by
    // the cheddar-to-emitc lowering and are NOT threaded here.
    // The Encoder is threaded into any function that does crypto OR encoding
    // (an encode-only function -- the preprocessing function -- needs it for
    // `encoder.Encode(...)`). The Context is threaded only into crypto
    // functions: the emitter emits the encode scale verbatim from the IR, so
    // encode-only functions no longer need the Context.
    auto hasCryptoOrEncode = [&](Operation* op) {
      return hasCryptoOps(op) || hasEncodeOps(op);
    };
    std::vector<std::pair<Type, OpPredicate>> evaluators = {
        {cheddar::ContextType::get(context), hasCryptoOps},
        {cheddar::EncoderType::get(context), hasCryptoOrEncode},
        {cheddar::UserInterfaceType::get(context), hasCryptoOps},
        {cheddar::EvalKeyType::get(context), hasCryptoOps},
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

    // Dynamically legal: func ops that have been converted. A function needs
    // a CHEDDAR context argument if it has crypto-typed args OR contains encode
    // ops (the encode-only preprocessing function takes only an encoder).
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      bool hasCheddarCtxArg =
          op.getFunctionType().getNumInputs() > 0 &&
          containsArgumentOfType<cheddar::ContextType, cheddar::EncoderType,
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
          mlir::isa<cheddar::ContextType>(*operandTypes.begin());
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
