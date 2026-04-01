#include "lib/Dialect/LWE/Conversions/LWEToCheddar/LWEToCheddar.h"

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
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
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

// Relinearize — needs the multiplication key
struct ConvertCKKSRelinOp : public OpConversionPattern<ckks::RelinearizeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ckks::RelinearizeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = getContextualArg<cheddar::ContextType>(op.getOperation());
    if (failed(ctx)) return ctx;
    auto ui = getContextualArg<cheddar::UserInterfaceType>(op.getOperation());
    if (failed(ui)) return ui;

    // Get the multiplication key from the UI
    auto multKey = cheddar::GetMultKeyOp::create(
        rewriter, op.getLoc(), cheddar::EvalKeyType::get(getContext()),
        ui.value());

    rewriter.replaceOpWithNewOp<cheddar::RelinearizeOp>(
        op, this->typeConverter->convertType(op.getOutput().getType()),
        ctx.value(), adaptor.getInput(), multKey);
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
struct ConvertCKKSRotateOp : public OpConversionPattern<ckks::RotateOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ckks::RotateOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = getContextualArg<cheddar::ContextType>(op.getOperation());
    if (failed(ctx)) return ctx;
    auto ui = getContextualArg<cheddar::UserInterfaceType>(op.getOperation());
    if (failed(ui)) return ui;

    auto shift = op.getStaticShiftAttr();
    if (!shift) {
      return op.emitOpError("CHEDDAR only supports static rotation shifts");
    }

    // Get rotation key for this distance
    auto rotKey = cheddar::GetRotKeyOp::create(
        rewriter, op.getLoc(), cheddar::EvalKeyType::get(getContext()),
        ui.value(), shift);

    rewriter.replaceOpWithNewOp<cheddar::HRotOp>(
        op, this->typeConverter->convertType(op.getOutput().getType()),
        ctx.value(), adaptor.getInput(), rotKey, shift);
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

    // TODO: compute target level from the mgmt attributes
    auto targetLevel = rewriter.getI64IntegerAttr(0);
    rewriter.replaceOpWithNewOp<cheddar::LevelDownOp>(
        op, this->typeConverter->convertType(op.getOutput().getType()),
        ctx.value(), adaptor.getInput(), targetLevel);
    return success();
  }
};

// Bootstrap
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

// Encode
struct ConvertLWEEncodeOp : public OpConversionPattern<lwe::RLWEEncodeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      lwe::RLWEEncodeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto encoder = getContextualArg<cheddar::EncoderType>(op.getOperation());
    if (failed(encoder)) return encoder;

    // TODO: derive level and scale from the encoding attributes
    auto level = rewriter.getI64IntegerAttr(0);
    auto scale = rewriter.getI64IntegerAttr(1ULL << 40);

    rewriter.replaceOpWithNewOp<cheddar::EncodeOp>(
        op, cheddar::PlaintextType::get(getContext()), encoder.value(),
        adaptor.getInput(), level, scale);
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

    rewriter.replaceOpWithNewOp<cheddar::DecodeOp>(
        op, op.getOutput().getType(), encoder.value(), adaptor.getInput());
    return success();
  }
};

// Orion linear transform
struct ConvertOrionLinearTransformOp
    : public OpConversionPattern<orion::LinearTransformOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      orion::LinearTransformOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = getContextualArg<cheddar::ContextType>(op.getOperation());
    if (failed(ctx)) return ctx;
    auto ui = getContextualArg<cheddar::UserInterfaceType>(op.getOperation());
    if (failed(ui)) return ui;

    auto evkMap = cheddar::GetEvkMapOp::create(
        rewriter, op.getLoc(), cheddar::EvkMapType::get(getContext()),
        ui.value());

    auto level = op.getOrionLevelAttr();
    auto bsgsRatio = op.getBsgsRatioAttr();
    int64_t logBsgsRatio = static_cast<int64_t>(bsgsRatio.getValueAsDouble());

    rewriter.replaceOpWithNewOp<cheddar::LinearTransformOp>(
        op, this->typeConverter->convertType(op.getResult().getType()),
        ctx.value(), adaptor.getInput(), evkMap, adaptor.getDiagonals(),
        op.getDiagonalIndicesAttr(), level,
        rewriter.getI64IntegerAttr(logBsgsRatio));
    return success();
  }
};

// Orion chebyshev
struct ConvertOrionChebyshevOp
    : public OpConversionPattern<orion::ChebyshevOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      orion::ChebyshevOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = getContextualArg<cheddar::ContextType>(op.getOperation());
    if (failed(ctx)) return ctx;
    auto ui = getContextualArg<cheddar::UserInterfaceType>(op.getOperation());
    if (failed(ui)) return ui;

    auto evkMap = cheddar::GetEvkMapOp::create(
        rewriter, op.getLoc(), cheddar::EvkMapType::get(getContext()),
        ui.value());

    rewriter.replaceOpWithNewOp<cheddar::EvalPolyOp>(
        op, this->typeConverter->convertType(op.getResult().getType()),
        ctx.value(), adaptor.getInput(), evkMap, op.getCoefficientsAttr());
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
    target.addIllegalOp<lwe::RLWEEncryptOp, lwe::RLWEDecryptOp,
                        lwe::RLWEEncodeOp, lwe::RLWEDecodeOp>();

    RewritePatternSet patterns(context);
    addStructuralConversionPatterns(typeConverter, patterns, target);

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

    // CHEDDAR context args to thread through functions
    std::vector<std::pair<Type, OpPredicate>> evaluators = {
        {cheddar::ContextType::get(context), hasCryptoOps},
        {cheddar::EncoderType::get(context),
         [&](Operation* op) { return hasCryptoOps(op) || hasEncodeOps(op); }},
        {cheddar::UserInterfaceType::get(context), hasCryptoOps},
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

    // LWE ops
    patterns.add<ConvertLWEEncodeOp, ConvertLWEDecodeOp, ConvertLWEEncryptOp,
                 ConvertLWEDecryptOp>(typeConverter, context);

    // Orion ops
    patterns.add<ConvertOrionLinearTransformOp, ConvertOrionChebyshevOp>(
        typeConverter, context);

    // Dynamically legal: func ops that have been converted
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      bool hasCheddarCtxArg = op.getFunctionType().getNumInputs() > 0 &&
                              containsArgumentOfType<cheddar::ContextType>(op);
      bool hasCryptoArg =
          containsArgumentOfDialect<lwe::LWEDialect, ckks::CKKSDialect>(op);
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody()) &&
             (!hasCryptoArg || hasCheddarCtxArg);
    });

    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      auto operandTypes = op.getCalleeType().getInputs();
      auto containsCryptoArg = llvm::any_of(operandTypes, [&](Type argType) {
        return DialectEqual<lwe::LWEDialect, ckks::CKKSDialect>()(
            &argType.getDialect());
      });
      auto hasCheddarCtxArg =
          !operandTypes.empty() &&
          mlir::isa<cheddar::ContextType>(*operandTypes.begin());
      return !containsCryptoArg || hasCheddarCtxArg;
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

// Include the generated pass definition
// (must be after the struct definition for the base class to find it)
