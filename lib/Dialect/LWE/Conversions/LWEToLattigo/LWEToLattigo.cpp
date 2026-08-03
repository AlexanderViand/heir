#include "lib/Dialect/LWE/Conversions/LWEToLattigo/LWEToLattigo.h"

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/BGV/IR/BGVOps.h"
#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Lattigo/IR/LattigoDialect.h"
#include "lib/Dialect/Lattigo/IR/LattigoOps.h"
#include "lib/Dialect/Lattigo/IR/LattigoTypes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/Orion/IR/OrionDialect.h"
#include "lib/Dialect/Orion/IR/OrionOps.h"
#include "lib/Dialect/Preprocessing/Conversions/Util.h"
#include "lib/Dialect/Preprocessing/IR/PreprocessingDialect.h"
#include "lib/Dialect/Preprocessing/IR/PreprocessingOps.h"
#include "lib/Dialect/Preprocessing/IR/PreprocessingTypes.h"
#include "lib/Utils/ConversionUtils.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/SmallSet.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Support/WalkResult.h"        // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "lwe-to-lattigo"

namespace mlir::heir::lwe {

class ToLattigoTypeConverter : public TypeConverter {
 public:
  ToLattigoTypeConverter(MLIRContext* ctx) {
    addConversion([](Type type) { return type; });
    addConversion([ctx](lwe::LWECiphertextType type) -> Type {
      return lattigo::RLWECiphertextType::get(ctx);
    });
    addConversion([ctx](lwe::LWEPlaintextType type) -> Type {
      return lattigo::RLWEPlaintextType::get(ctx);
    });
    addConversion([ctx](lwe::LWEPublicKeyType type) -> Type {
      return lattigo::RLWEPublicKeyType::get(ctx);
    });
    addConversion([ctx](lwe::LWESecretKeyType type) -> Type {
      return lattigo::RLWESecretKeyType::get(ctx);
    });
    addConversion([this](RankedTensorType type) -> Type {
      return RankedTensorType::get(type.getShape(),
                                   this->convertType(type.getElementType()));
    });
    addConversion([this](MemRefType type) -> Type {
      return MemRefType::get(type.getShape(),
                             this->convertType(type.getElementType()));
    });
    addConversion([this](preprocessing::PreprocessingStorageType type) -> Type {
      return preprocessing::convertStorageElementTypes(type, this);
    });
  }
};

namespace {
template <typename EvaluatorType>
FailureOr<Value> getContextualEvaluator(Operation* op) {
  auto result = getContextualArgFromFunc<EvaluatorType>(op);
  if (failed(result)) {
    return op->emitOpError()
           << "Found RLWE op in a function without a public "
              "key argument. Did the AddEvaluatorArg pattern fail to run?";
  }
  return result.value();
}

FailureOr<Value> getContextualEvaluator(Operation* op, Type type) {
  return getContextualArgFromFunc(op, type);
}

// Find the unique operation in the current func whose result has type Ty or
// return Failure.
template <typename Ty>
FailureOr<TypedValue<Ty>> findUniqueOpResult(Operation* op) {
  TypedValue<Ty> foundValue;
  func::FuncOp funcOp = op->getParentOfType<func::FuncOp>();
  auto result = funcOp->walk([&](Operation* innerOp) {
    for (auto result : innerOp->getResults()) {
      if (mlir::isa<Ty>(result.getType())) {
        foundValue = cast<TypedValue<Ty>>(result);
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    return foundValue;
  }
  return failure();
}

// NOTE: we can not use containsDialect
// for FuncOp declaration, which does not have a body
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

bool containsBootstrap(Operation* op) {
  auto funcOp = dyn_cast<func::FuncOp>(op);
  if (!funcOp) {
    return false;
  }
  auto result = walkFuncAndCallees(funcOp, [&](Operation* op) {
    if (isa<ckks::BootstrapOp, lattigo::CKKSBootstrapOp>(op)) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result.wasInterrupted();
}

// Walk all `ckks.bootstrap` ops in the func and its callees, collecting the
// distinct logSlots values requested. A missing `logSlots` attribute is
// represented by std::nullopt (i.e. "default / full-slot bootstrapper").
// The returned vector is sorted: nullopt first (default), then ascending.
SmallVector<std::optional<int>> collectDistinctBootstrapLogSlots(
    func::FuncOp funcOp) {
  bool hasDefault = false;
  llvm::SmallSet<int, 4> sparseSlots;
  walkFuncAndCallees(funcOp, [&](Operation* op) {
    if (auto bootOp = dyn_cast<ckks::BootstrapOp>(op)) {
      if (auto attr = bootOp.getLogSlotsAttr()) {
        sparseSlots.insert(static_cast<int>(attr.getInt()));
      } else {
        hasDefault = true;
      }
    }
    return WalkResult::advance();
  });
  SmallVector<std::optional<int>> result;
  if (hasDefault) {
    result.push_back(std::nullopt);
  }
  SmallVector<int> sortedSparse(sparseSlots.begin(), sparseSlots.end());
  llvm::sort(sortedSparse);
  for (int s : sortedSparse) {
    result.push_back(s);
  }
  return result;
}

// Find the bootstrap-evaluator block arg in `funcOp` whose
// `lattigo.bootstrap_log_slots` arg-attr matches `targetLogSlots`. Returns
// nullptr if no match is found.
Value findBootstrapEvaluatorArg(func::FuncOp funcOp,
                                std::optional<int> targetLogSlots) {
  for (unsigned argIdx = 0; argIdx < funcOp.getNumArguments(); ++argIdx) {
    auto blockArg = funcOp.getArgument(argIdx);
    if (!isa<lattigo::CKKSBootstrappingEvaluatorType>(blockArg.getType()))
      continue;
    std::optional<int> argLogSlots;
    if (auto attr = funcOp.getArgAttrOfType<IntegerAttr>(
            argIdx, lattigo::kBootstrapLogSlotsAttrName)) {
      argLogSlots = static_cast<int>(attr.getInt());
    }
    if (argLogSlots == targetLogSlots) {
      return blockArg;
    }
  }
  return {};
}

bool containsEncode(Operation* op) {
  auto funcOp = dyn_cast<func::FuncOp>(op);
  if (!funcOp) {
    return false;
  }
  auto result = walkFuncAndCallees(funcOp, [&](Operation* op) {
    if (isa<lwe::RLWEEncodeOp>(op)) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result.wasInterrupted();
}

struct AddEvaluatorArg : public OpConversionPattern<func::FuncOp> {
  AddEvaluatorArg(mlir::MLIRContext* context,
                  const std::vector<std::pair<Type, OpPredicate>>& evaluators)
      : OpConversionPattern<func::FuncOp>(context, /* benefit= */ 2),
        evaluators(evaluators) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::FuncOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    SmallVector<Type, 4> selectedEvaluators;
    SmallVector<DictionaryAttr, 4> selectedArgAttrs;
    LLVM_DEBUG(llvm::dbgs()
               << "AddEvaluatorArg for func " << op.getName() << "\n");

    for (const auto& evaluator : evaluators) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Checking if evaluator should be added of type: "
                 << evaluator.first << "\n");
      auto predicate = evaluator.second;
      if (!predicate(op)) {
        continue;
      }
      // Bootstrap evaluator: one arg per distinct logSlots value found in
      // the func's `ckks.bootstrap` ops. Each arg is tagged with its
      // `lattigo.bootstrap_log_slots` so later passes (ConvertBootstrapOp,
      // ConfigureCryptoContext) can disambiguate by sparse slot count.
      if (isa<lattigo::CKKSBootstrappingEvaluatorType>(evaluator.first)) {
        auto logSlotsList = collectDistinctBootstrapLogSlots(op);
        if (logSlotsList.empty()) {
          // Predicate said yes (so containsBootstrap was true) but we found
          // nothing; fall back to a single default arg.
          logSlotsList.push_back(std::nullopt);
        }
        for (std::optional<int> ls : logSlotsList) {
          LLVM_DEBUG(llvm::dbgs()
                     << "Adding bootstrap evaluator with logSlots="
                     << (ls ? std::to_string(*ls) : std::string("<default>"))
                     << "\n");
          selectedEvaluators.push_back(evaluator.first);
          if (ls.has_value()) {
            NamedAttribute named(
                rewriter.getStringAttr(lattigo::kBootstrapLogSlotsAttrName),
                rewriter.getI64IntegerAttr(*ls));
            selectedArgAttrs.push_back(rewriter.getDictionaryAttr({named}));
          } else {
            selectedArgAttrs.push_back(nullptr);
          }
        }
        continue;
      }
      LLVM_DEBUG(llvm::dbgs()
                 << "Adding evaluator of type: " << evaluator.first << "\n");
      selectedEvaluators.push_back(evaluator.first);
      selectedArgAttrs.push_back(nullptr);
    }

    if (selectedEvaluators.empty()) {
      return rewriter.notifyMatchFailure(op, "no evaluator needed");
    }

    // Insert all argument at the beginning
    // NOTE: arguments with identical index will
    // appear in the same order that they were listed.
    SmallVector<Location> argLocs(selectedEvaluators.size(), op.getLoc());

    rewriter.modifyOpInPlace(op, [&] {
      SmallVector<unsigned> argIndices(selectedEvaluators.size(), 0);
      (void)op.insertArguments(argIndices, selectedEvaluators, selectedArgAttrs,
                               argLocs);
    });
    return success();
  }

 private:
  std::vector<std::pair<Type, OpPredicate>> evaluators;
};

template <typename KeyType>
struct RemoveKeyArg : public OpRewritePattern<func::FuncOp> {
  RemoveKeyArg(mlir::MLIRContext* context)
      : OpRewritePattern<func::FuncOp>(context, /* benefit= */ 2) {}

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter& rewriter) const override {
    ::llvm::BitVector argsToErase(op.getNumArguments());

    for (auto i = 0; i != op.getNumArguments(); ++i) {
      if (mlir::isa<KeyType>(op.getArgumentTypes()[i])) {
        argsToErase.set(i);
      }
    }

    if (argsToErase.none()) {
      return rewriter.notifyMatchFailure(op, "no key arguments to erase");
    }

    rewriter.modifyOpInPlace(op, [&] { (void)op.eraseArguments(argsToErase); });
    return success();
  }
};

struct ConvertFuncCallOp : public OpConversionPattern<func::CallOp> {
  ConvertFuncCallOp(mlir::MLIRContext* context,
                    const std::vector<std::pair<Type, OpPredicate>>& evaluators)
      : OpConversionPattern<func::CallOp>(context), evaluators(evaluators) {}

  using OpConversionPattern<func::CallOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::CallOp op, typename func::CallOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    LLVM_DEBUG(llvm::dbgs()
               << "ConvertFuncCallOp for func call "
               << op.getCalleeAttr().getValue() << " with signature "
               << op.getCalleeType() << "\n");
    auto funcOp = getCalledFunction(op);
    if (failed(funcOp)) {
      return rewriter.notifyMatchFailure(op, "could not find callee function");
    }
    SmallVector<Value> selectedevaluatorsValues;
    for (const auto& evaluator : evaluators) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Checking if evaluator should be added of type: "
                 << evaluator.first << "\n");
      // Bootstrap evaluator: the callee may take multiple, one per distinct
      // sparse `logSlots`. Forward one matching value per callee arg by
      // matching `lattigo.bootstrap_log_slots`. (Use argument types — not
      // getArgument — so this works on callee declarations too.)
      if (isa<lattigo::CKKSBootstrappingEvaluatorType>(evaluator.first)) {
        func::FuncOp callee = funcOp.value();
        auto callerFunc = op->getParentOfType<func::FuncOp>();
        auto calleeArgTypes = callee.getArgumentTypes();
        if (!callerFunc) continue;
        for (unsigned argIdx = 0; argIdx < calleeArgTypes.size(); ++argIdx) {
          if (calleeArgTypes[argIdx] != evaluator.first) continue;
          std::optional<int> calleeLogSlots;
          if (auto attr = callee.getArgAttrOfType<IntegerAttr>(
                  argIdx, lattigo::kBootstrapLogSlotsAttrName)) {
            calleeLogSlots = static_cast<int>(attr.getInt());
          }
          Value match = findBootstrapEvaluatorArg(callerFunc, calleeLogSlots);
          if (!match) {
            return rewriter.notifyMatchFailure(
                op, "caller has no bootstrap evaluator matching callee arg");
          }
          LLVM_DEBUG(llvm::dbgs()
                     << "Forwarding bootstrap evaluator logSlots="
                     << (calleeLogSlots ? std::to_string(*calleeLogSlots)
                                        : std::string("<default>"))
                     << "\n");
          selectedevaluatorsValues.push_back(match);
        }
        continue;
      }

      auto result = getContextualEvaluator(op.getOperation(), evaluator.first);
      // filter out non-existent evaluators
      if (failed(result)) {
        continue;
      }
      // filter for only the required evaluators
      if (!llvm::any_of(funcOp.value().getArgumentTypes(), [&](Type argType) {
            return evaluator.first == argType;
          })) {
        continue;
      }
      LLVM_DEBUG(llvm::dbgs()
                 << "Adding evaluator of type: " << evaluator.first << "\n");
      selectedevaluatorsValues.push_back(result.value());
    }

    auto callee = op.getCallee();
    auto operands = adaptor.getOperands();
    auto resultTypes = op.getResultTypes();

    SmallVector<Value> newOperands;
    for (auto evaluator : selectedevaluatorsValues) {
      newOperands.push_back(evaluator);
    }
    for (auto operand : operands) {
      newOperands.push_back(operand);
    }

    SmallVector<NamedAttribute> dialectAttrs(op->getDialectAttrs());
    rewriter
        .replaceOpWithNewOp<func::CallOp>(op, callee, resultTypes, newOperands)
        ->setDialectAttrs(dialectAttrs);
    return success();
  }

 private:
  std::vector<std::pair<Type, OpPredicate>> evaluators;
};

template <typename KeyType>
struct RemoveKeyArgForFuncCall : public OpConversionPattern<func::CallOp> {
  RemoveKeyArgForFuncCall(mlir::MLIRContext* context)
      : OpConversionPattern<func::CallOp>(context) {}

  using OpConversionPattern<func::CallOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::CallOp op, typename func::CallOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto callee = op.getCallee();
    auto operands = adaptor.getOperands();
    auto resultTypes = op.getResultTypes();

    SmallVector<Value> newOperands;
    for (auto operand : operands) {
      if (!mlir::isa<KeyType>(operand.getType())) {
        newOperands.push_back(operand);
      }
    }

    SmallVector<NamedAttribute> dialectAttrs(op->getDialectAttrs());
    rewriter
        .replaceOpWithNewOp<func::CallOp>(op, callee, resultTypes, newOperands)
        ->setDialectAttrs(dialectAttrs);
    return success();
  }
};

template <typename EvaluatorType, typename UnaryOp, typename LattigoUnaryOp>
struct ConvertRlweUnaryOp : public OpConversionPattern<UnaryOp> {
  using OpConversionPattern<UnaryOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      UnaryOp op, typename UnaryOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    FailureOr<Value> result =
        getContextualEvaluator<EvaluatorType>(op.getOperation());
    if (failed(result)) return result;

    Value evaluator = result.value();
    rewriter.replaceOp(
        op, LattigoUnaryOp::create(
                rewriter, op.getLoc(),
                this->typeConverter->convertType(op.getOutput().getType()),
                evaluator, adaptor.getInput()));
    return success();
  }
};

template <typename EvaluatorType, typename BinOp, typename LattigoBinOp>
struct ConvertRlweBinOp : public OpConversionPattern<BinOp> {
  using OpConversionPattern<BinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      BinOp op, typename BinOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    FailureOr<Value> result =
        getContextualEvaluator<EvaluatorType>(op.getOperation());
    if (failed(result)) return result;

    Value evaluator = result.value();
    rewriter.replaceOpWithNewOp<LattigoBinOp>(
        op, this->typeConverter->convertType(op.getOutput().getType()),
        evaluator, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

// Lattigo API enforces ciphertext, plaintext ordering.
template <typename EvaluatorType, typename PlainOp, typename LattigoPlainOp>
struct ConvertRlweCommutativePlainOp : public OpConversionPattern<PlainOp> {
  using OpConversionPattern<PlainOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      PlainOp op, typename PlainOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    FailureOr<Value> result =
        getContextualEvaluator<EvaluatorType>(op.getOperation());
    if (failed(result)) return result;

    Value ciphertext = adaptor.getLhs();
    Value plaintext = adaptor.getRhs();
    if (!isa<lattigo::RLWECiphertextType>(adaptor.getLhs().getType())) {
      ciphertext = adaptor.getRhs();
      plaintext = adaptor.getLhs();
    }

    Value evaluator = result.value();
    rewriter.replaceOpWithNewOp<LattigoPlainOp>(
        op, this->typeConverter->convertType(op.getOutput().getType()),
        evaluator, ciphertext, plaintext);
    return success();
  }
};

// Lattigo API enforces ciphertext, plaintext ordering.
template <typename EvaluatorType, typename PlainOp, typename LattigoPlainOp,
          typename LattigoAddOp, typename LattigoNegateOp>
struct ConvertRlweSubPlainOp : public OpConversionPattern<PlainOp> {
  using OpConversionPattern<PlainOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      PlainOp op, typename PlainOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    FailureOr<Value> result =
        getContextualEvaluator<EvaluatorType>(op.getOperation());
    if (failed(result)) return result;

    Value evaluator = result.value();
    if (isa<lattigo::RLWECiphertextType>(adaptor.getLhs().getType())) {
      // Lattigo API enforces ciphertext, plaintext ordering, so we can use
      // LattigoPlainOp directly.
      Value ciphertext = adaptor.getLhs();
      Value plaintext = adaptor.getRhs();
      rewriter.replaceOpWithNewOp<LattigoPlainOp>(
          op, this->typeConverter->convertType(op.getOutput().getType()),
          evaluator, ciphertext, plaintext);
      return success();
    }

    // handle plaintext - ciphertext using (-ciphertext) + plaintext
    Value plaintext = adaptor.getLhs();
    Value ciphertext = adaptor.getRhs();

    auto negated = LattigoNegateOp::create(
        rewriter, op.getLoc(),
        this->typeConverter->convertType(op.getOutput().getType()), evaluator,
        ciphertext);
    rewriter.replaceOpWithNewOp<LattigoAddOp>(
        op, this->typeConverter->convertType(op.getOutput().getType()),
        evaluator, negated, plaintext);
    return success();
  }
};

template <typename EvaluatorType, typename RlweRotateOp,
          typename LattigoRotateOp>
struct ConvertRlweRotateOp : public OpConversionPattern<RlweRotateOp> {
  ConvertRlweRotateOp(mlir::MLIRContext* context)
      : OpConversionPattern<RlweRotateOp>(context) {}

  using OpConversionPattern<RlweRotateOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      RlweRotateOp op, typename RlweRotateOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    FailureOr<Value> result =
        getContextualEvaluator<EvaluatorType>(op.getOperation());
    if (failed(result))
      return rewriter.notifyMatchFailure(op,
                                         "Failed to get contextual evaluator");

    Value evaluator = result.value();
    // Use dynamic_shift SSA value if available, otherwise fall back to
    // static_shift attribute (static case for backward compatibility)
    Value dynamicShift = adaptor.getDynamicShift();
    IntegerAttr staticShift = op.getStaticShiftAttr();
    if (!staticShift && !dynamicShift) {
      return rewriter.notifyMatchFailure(
          op, "rotate op must have either static or dynamic shift");
    }

    if (dynamicShift) {
      rewriter.replaceOp(
          op, LattigoRotateOp::create(
                  rewriter, op.getLoc(),
                  this->typeConverter->convertType(op.getOutput().getType()),
                  evaluator, adaptor.getInput(), dynamicShift,
                  /*static_shift=*/nullptr));
    } else {
      rewriter.replaceOp(
          op, LattigoRotateOp::create(
                  rewriter, op.getLoc(),
                  this->typeConverter->convertType(op.getOutput().getType()),
                  evaluator, adaptor.getInput(), /*dynamic_shift=*/nullptr,
                  staticShift));
    }
    return success();
  }
};

// CKKS bootstrap lowering: pick the bootstrap-evaluator block arg whose
// `lattigo.bootstrap_log_slots` arg-attr matches this op's `logSlots`
// (or no attr if this op has no logSlots, i.e. the default/full-slot
// bootstrapper).
struct ConvertCKKSBootstrapOp : public OpConversionPattern<ckks::BootstrapOp> {
  using OpConversionPattern<ckks::BootstrapOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ckks::BootstrapOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto funcOp = op->getParentOfType<func::FuncOp>();
    if (!funcOp) {
      return rewriter.notifyMatchFailure(op, "bootstrap op not in a function");
    }

    std::optional<int> targetLogSlots;
    if (auto attr = op.getLogSlotsAttr()) {
      targetLogSlots = static_cast<int>(attr.getInt());
    }

    Value evaluator = findBootstrapEvaluatorArg(funcOp, targetLogSlots);
    if (!evaluator) {
      return rewriter.notifyMatchFailure(
          op, "no bootstrap evaluator arg matching this op's logSlots");
    }

    rewriter.replaceOpWithNewOp<lattigo::CKKSBootstrapOp>(
        op, this->typeConverter->convertType(op.getOutput().getType()),
        evaluator, adaptor.getInput());
    return success();
  }
};

template <typename EvaluatorType, typename LevelReduceOp,
          typename LattigoLevelReduceOp>
struct ConvertRlweLevelReduceOp : public OpConversionPattern<LevelReduceOp> {
  using OpConversionPattern<LevelReduceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      LevelReduceOp op, typename LevelReduceOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    FailureOr<Value> result =
        getContextualEvaluator<EvaluatorType>(op.getOperation());
    if (failed(result)) return result;

    Value evaluator = result.value();
    rewriter.replaceOp(
        op, LattigoLevelReduceOp::create(
                rewriter, op.getLoc(),
                this->typeConverter->convertType(op.getOutput().getType()),
                evaluator, adaptor.getInput(), op.getLevelToDrop()));
    return success();
  }
};

template <typename EvaluatorType, typename ParamType, typename EncodeOp,
          typename LattigoEncodeOp, typename AllocOp>
struct ConvertRlweEncodeOp : public OpConversionPattern<EncodeOp> {
  using OpConversionPattern<EncodeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      EncodeOp op, typename EncodeOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    FailureOr<Value> result =
        getContextualEvaluator<EvaluatorType>(op.getOperation());
    if (failed(result)) return result;
    Value evaluator = result.value();

    FailureOr<Value> result2 =
        getContextualEvaluator<ParamType>(op.getOperation());
    if (failed(result2)) return result2;
    Value params = result2.value();

    Value input = adaptor.getInput();
    auto alloc = AllocOp::create(
        rewriter, op.getLoc(),
        this->typeConverter->convertType(op.getOutput().getType()), params);

    auto encoding = op.getEncoding();
    int64_t scale = lwe::getScalingFactorFromEncodingAttr(encoding);

    SmallVector<NamedAttribute> dialectAttrs(op->getDialectAttrs());
    rewriter
        .replaceOpWithNewOp<LattigoEncodeOp>(
            op, this->typeConverter->convertType(op.getOutput().getType()),
            evaluator, input, alloc, rewriter.getI64IntegerAttr(scale))
        ->setDialectAttrs(dialectAttrs);
    return success();
  }
};

template <typename EvaluatorType, typename DecodeOp, typename LattigoDecodeOp,
          typename AllocOp>
struct ConvertRlweDecodeOp : public OpConversionPattern<DecodeOp> {
  using OpConversionPattern<DecodeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      DecodeOp op, typename DecodeOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    FailureOr<Value> result =
        getContextualEvaluator<EvaluatorType>(op.getOperation());
    if (failed(result)) return result;
    Value evaluator = result.value();

    // With the new layout system, decoding for RLWE will always produce a
    // tensor type.
    auto outputType = op.getOutput().getType();
    RankedTensorType outputTensorType = dyn_cast<RankedTensorType>(outputType);

    auto zeroAttr = rewriter.getZeroAttr(outputTensorType);
    if (!zeroAttr) {
      return rewriter.notifyMatchFailure(op, "Unsupported type for lowering");
    }
    auto alloc =
        AllocOp::create(rewriter, op.getLoc(), outputTensorType, zeroAttr);

    auto decodeOp =
        LattigoDecodeOp::create(rewriter, op.getLoc(), outputTensorType,
                                evaluator, adaptor.getInput(), alloc);

    rewriter.replaceOp(op, decodeOp.getResult());
    return success();
  }
};

// Orion conversions
struct ConvertOrionLinearTransformOp
    : public OpConversionPattern<orion::LinearTransformOp> {
  using OpConversionPattern<orion::LinearTransformOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      orion::LinearTransformOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    FailureOr<Value> evaluatorResult =
        getContextualEvaluator<lattigo::CKKSEvaluatorType>(op.getOperation());
    if (failed(evaluatorResult)) {
      return op.emitOpError() << "CKKS evaluator not found in function context";
    }
    Value evaluator = evaluatorResult.value();

    FailureOr<Value> encoderResult =
        getContextualEvaluator<lattigo::CKKSEncoderType>(op.getOperation());
    if (failed(encoderResult)) {
      return op.emitOpError() << "CKKS encoder not found in function context";
    }
    Value encoder = encoderResult.value();

    // orion.linear_transform carries a *linear* baby/giant ratio (e.g. 2.0);
    // Lattigo's LogBabyStepGiantStepRatio wants its log2.
    double bsgsRatio = op.getBsgsRatioAttr().getValueAsDouble();
    int64_t logBsgsRatio =
        bsgsRatio >= 1 ? std::llround(std::log2(bsgsRatio)) : 0;
    auto logBsgsRatioAttr = rewriter.getI64IntegerAttr(logBsgsRatio);

    rewriter.replaceOpWithNewOp<lattigo::CKKSLinearTransformOp>(
        op, this->typeConverter->convertType(op.getResult().getType()),
        evaluator, encoder, adaptor.getInput(), adaptor.getDiagonals(),
        adaptor.getDiagonalIndices(), op.getOrionLevelAttr(), logBsgsRatioAttr);

    return success();
  }
};

struct ConvertOrionChebyshevOp
    : public OpConversionPattern<orion::ChebyshevOp> {
  using OpConversionPattern<orion::ChebyshevOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      orion::ChebyshevOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "Lowering Orion ChebyshevOp\n");
    // Get or create the polynomial evaluator from the function context
    FailureOr<Value> evaluatorResult =
        findUniqueOpResult<lattigo::CKKSPolynomialEvaluatorType>(
            op.getOperation());
    Value polyEvaluator;
    if (failed(evaluatorResult)) {
      LLVM_DEBUG(llvm::dbgs() << "Creating new CKKS polynomial evaluator\n");
      FailureOr<Value> evaluatorResult =
          getContextualEvaluator<lattigo::CKKSEvaluatorType>(op.getOperation());
      if (failed(evaluatorResult)) {
        return rewriter.notifyMatchFailure(
            op, "CKKS evaluator not found in function context");
      }
      Value evaluator = evaluatorResult.value();

      FailureOr<Value> result2 =
          getContextualEvaluator<lattigo::CKKSParameterType>(op.getOperation());
      if (failed(result2))
        return rewriter.notifyMatchFailure(
            op, "Failed to get contextual CKKS parameters");
      Value params = result2.value();

      // Insert op at start of func so it's easy to find later
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(
            &op->getParentOfType<func::FuncOp>().getBody().front());
        auto evaluatorOp = lattigo::CKKSNewPolynomialEvaluatorOp::create(
            rewriter, op.getLoc(),
            lattigo::CKKSPolynomialEvaluatorType::get(rewriter.getContext()),
            params, evaluator);
        polyEvaluator = evaluatorOp.getResult();
      }
    } else {
      polyEvaluator = evaluatorResult.value();
    }

    // Target scale: honor the op's intended output scale if present, otherwise
    // fall back to the default scale (2^logDefaultScale). Orion's
    // composite-sign chain sets the last polynomial's output scale to `ql` so
    // the following ciphertext-multiply's rescale is exact; forcing the default
    // scale there breaks the scale contract.
    ckks::SchemeParamAttr schemeParams =
        cast<ckks::SchemeParamAttr>(getSchemeParamAttr(op));
    IntegerAttr targetScale;
    if (IntegerAttr outputScale = op.getOutputScaleAttr()) {
      targetScale =
          rewriter.getIntegerAttr(rewriter.getI64Type(), outputScale.getInt());
      LLVM_DEBUG(llvm::dbgs() << "Using orion output scale: "
                              << targetScale.getInt() << "\n");
    } else {
      targetScale = rewriter.getIntegerAttr(
          rewriter.getI64Type(), 1L << schemeParams.getLogDefaultScale());
      LLVM_DEBUG(llvm::dbgs()
                 << "Using default scale: " << targetScale.getInt() << "\n");
    }

    // Compute the multiplicative depth consumed by the polynomial evaluation
    // from the orion op's input/result modulus chains. The lattigo ciphertext
    // type is opaque w.r.t. level, so carry the drop as an attribute and via
    // the ReducesLevelOpInterface so LevelAnalysis (and the in-place allocation
    // it drives) sees the level drop. Mirrors LWEToCheddar's ConvertOrion
    // ChebyshevOp, which reads input level and output level from these chains.
    auto inTy = dyn_cast<lwe::LWECiphertextType>(op.getInput().getType());
    auto resTy = dyn_cast<lwe::LWECiphertextType>(op.getResult().getType());
    if (!inTy || !inTy.getModulusChain() || !resTy || !resTy.getModulusChain())
      return op.emitOpError()
             << "cannot lower orion.chebyshev to lattigo.ckks.chebyshev: "
                "input/result ciphertext has no modulus chain (run the "
                "upstream CKKS level analysis first)";
    int64_t inputLevel = inTy.getModulusChain().getCurrent();
    int64_t outputLevel = resTy.getModulusChain().getCurrent();
    int64_t levelToDrop = inputLevel - outputLevel;
    if (levelToDrop < 0)
      return op.emitOpError()
             << "orion.chebyshev result level (" << outputLevel
             << ") is above its input level (" << inputLevel
             << "); polynomial evaluation cannot raise the level";
    IntegerAttr levelToDropAttr =
        rewriter.getIntegerAttr(rewriter.getI64Type(), levelToDrop);

    auto domainAttr =
        rewriter.getDenseF64ArrayAttr({op.getDomainStart().getValueAsDouble(),
                                       op.getDomainEnd().getValueAsDouble()});
    auto chebyshevOp = lattigo::CKKSChebyshevOp::create(
        rewriter, op.getLoc(), adaptor.getInput().getType(), polyEvaluator,
        adaptor.getInput(), adaptor.getCoefficients(), targetScale, domainAttr,
        levelToDropAttr);
    rewriter.replaceOp(op, chebyshevOp.getResult());

    return success();
  }
};

}  // namespace

// BGV
using ConvertBGVAddOp = ConvertRlweBinOp<lattigo::BGVEvaluatorType, lwe::RAddOp,
                                         lattigo::BGVAddNewOp>;
using ConvertBGVSubOp = ConvertRlweBinOp<lattigo::BGVEvaluatorType, lwe::RSubOp,
                                         lattigo::BGVSubNewOp>;
using ConvertBGVMulOp = ConvertRlweBinOp<lattigo::BGVEvaluatorType, lwe::RMulOp,
                                         lattigo::BGVMulNewOp>;
using ConvertBGVAddPlainOp =
    ConvertRlweCommutativePlainOp<lattigo::BGVEvaluatorType, lwe::RAddPlainOp,
                                  lattigo::BGVAddNewOp>;
using ConvertBGVSubPlainOp =
    ConvertRlweSubPlainOp<lattigo::BGVEvaluatorType, lwe::RSubPlainOp,
                          lattigo::BGVSubNewOp, lattigo::BGVAddNewOp,
                          lattigo::RLWENegateNewOp>;
using ConvertBGVMulPlainOp =
    ConvertRlweCommutativePlainOp<lattigo::BGVEvaluatorType, lwe::RMulPlainOp,
                                  lattigo::BGVMulNewOp>;

using ConvertBGVRelinOp =
    ConvertRlweUnaryOp<lattigo::BGVEvaluatorType, bgv::RelinearizeOp,
                       lattigo::BGVRelinearizeNewOp>;
using ConvertBGVModulusSwitchOp =
    ConvertRlweUnaryOp<lattigo::BGVEvaluatorType, bgv::ModulusSwitchOp,
                       lattigo::BGVRescaleNewOp>;

using ConvertBGVRotateColumnsOp =
    ConvertRlweRotateOp<lattigo::BGVEvaluatorType, bgv::RotateColumnsOp,
                        lattigo::BGVRotateColumnsNewOp>;

using ConvertBGVEncryptOp =
    ConvertRlweUnaryOp<lattigo::RLWEEncryptorType, lwe::RLWEEncryptOp,
                       lattigo::RLWEEncryptOp>;
using ConvertBGVDecryptOp =
    ConvertRlweUnaryOp<lattigo::RLWEDecryptorType, lwe::RLWEDecryptOp,
                       lattigo::RLWEDecryptOp>;
using ConvertBGVEncodeOp =
    ConvertRlweEncodeOp<lattigo::BGVEncoderType, lattigo::BGVParameterType,
                        lwe::RLWEEncodeOp, lattigo::BGVEncodeOp,
                        lattigo::BGVNewPlaintextOp>;
using ConvertBGVDecodeOp =
    ConvertRlweDecodeOp<lattigo::BGVEncoderType, lwe::RLWEDecodeOp,
                        lattigo::BGVDecodeOp, arith::ConstantOp>;

using ConvertBGVLevelReduceOp =
    ConvertRlweLevelReduceOp<lattigo::BGVEvaluatorType, bgv::LevelReduceOp,
                             lattigo::RLWEDropLevelNewOp>;

// CKKS
using ConvertCKKSAddOp = ConvertRlweBinOp<lattigo::CKKSEvaluatorType,
                                          lwe::RAddOp, lattigo::CKKSAddNewOp>;
using ConvertCKKSSubOp = ConvertRlweBinOp<lattigo::CKKSEvaluatorType,
                                          lwe::RSubOp, lattigo::CKKSSubNewOp>;
using ConvertCKKSMulOp = ConvertRlweBinOp<lattigo::CKKSEvaluatorType,
                                          lwe::RMulOp, lattigo::CKKSMulNewOp>;
using ConvertCKKSAddPlainOp =
    ConvertRlweCommutativePlainOp<lattigo::CKKSEvaluatorType, lwe::RAddPlainOp,
                                  lattigo::CKKSAddNewOp>;
using ConvertCKKSSubPlainOp =
    ConvertRlweSubPlainOp<lattigo::CKKSEvaluatorType, lwe::RSubPlainOp,
                          lattigo::CKKSSubNewOp, lattigo::CKKSAddNewOp,
                          lattigo::RLWENegateNewOp>;
using ConvertCKKSMulPlainOp =
    ConvertRlweCommutativePlainOp<lattigo::CKKSEvaluatorType, lwe::RMulPlainOp,
                                  lattigo::CKKSMulNewOp>;

using ConvertCKKSRelinOp =
    ConvertRlweUnaryOp<lattigo::CKKSEvaluatorType, ckks::RelinearizeOp,
                       lattigo::CKKSRelinearizeNewOp>;
using ConvertCKKSModulusSwitchOp =
    ConvertRlweUnaryOp<lattigo::CKKSEvaluatorType, ckks::RescaleOp,
                       lattigo::CKKSRescaleNewOp>;

using ConvertCKKSRotateOp =
    ConvertRlweRotateOp<lattigo::CKKSEvaluatorType, ckks::RotateOp,
                        lattigo::CKKSRotateNewOp>;

using ConvertCKKSEncryptOp =
    ConvertRlweUnaryOp<lattigo::RLWEEncryptorType, lwe::RLWEEncryptOp,
                       lattigo::RLWEEncryptOp>;
using ConvertCKKSDecryptOp =
    ConvertRlweUnaryOp<lattigo::RLWEDecryptorType, lwe::RLWEDecryptOp,
                       lattigo::RLWEDecryptOp>;
using ConvertCKKSEncodeOp =
    ConvertRlweEncodeOp<lattigo::CKKSEncoderType, lattigo::CKKSParameterType,
                        lwe::RLWEEncodeOp, lattigo::CKKSEncodeOp,
                        lattigo::CKKSNewPlaintextOp>;
using ConvertCKKSDecodeOp =
    ConvertRlweDecodeOp<lattigo::CKKSEncoderType, lwe::RLWEDecodeOp,
                        lattigo::CKKSDecodeOp, arith::ConstantOp>;

using ConvertCKKSLevelReduceOp =
    ConvertRlweLevelReduceOp<lattigo::CKKSEvaluatorType, ckks::LevelReduceOp,
                             lattigo::RLWEDropLevelNewOp>;

#define GEN_PASS_DEF_LWETOLATTIGO
#include "lib/Dialect/LWE/Conversions/LWEToLattigo/LWEToLattigo.h.inc"

struct LWEToLattigo : public impl::LWEToLattigoBase<LWEToLattigo> {
  // See https://github.com/llvm/llvm-project/pull/127772
  // During dialect conversion, the attribute of the func::CallOp is not
  // preserved. We save the dialect attributes of func::CallOp before
  // conversion and restore them after conversion.
  //
  // Note that this is not safe as after conversion the order of func::CallOp
  // may change. However, this is the best we can do for now as we do not have
  // a map from the old func::CallOp to the new func::CallOp.
  SmallVector<SmallVector<NamedAttribute>> funcCallOpDialectAttrs;

  void saveFuncCallOpDialectAttrs() {
    funcCallOpDialectAttrs.clear();
    auto* module = getOperation();
    module->walk([&](func::CallOp callOp) {
      SmallVector<NamedAttribute> dialectAttrs;
      for (auto namedAttr : callOp->getDialectAttrs()) {
        dialectAttrs.push_back(namedAttr);
      }
      funcCallOpDialectAttrs.push_back(dialectAttrs);
    });
  }

  void restoreFuncCallOpDialectAttrs() {
    auto* module = getOperation();
    auto* funcCallOpDialectAttrsIter = funcCallOpDialectAttrs.begin();
    module->walk([&](func::CallOp callOp) {
      callOp->setDialectAttrs(*funcCallOpDialectAttrsIter);
      ++funcCallOpDialectAttrsIter;
    });
  }

  void runOnOperation() override {
    // Save the dialect attributes of func::CallOp before conversion.
    saveFuncCallOpDialectAttrs();

    MLIRContext* context = &getContext();
    auto* module = getOperation();
    ToLattigoTypeConverter typeConverter(context);

    ConversionTarget target(*context);
    target.addLegalDialect<lattigo::LattigoDialect>();
    target.addIllegalDialect<bgv::BGVDialect, ckks::CKKSDialect,
                             orion::OrionDialect>();
    target.addDynamicallyLegalDialect<preprocessing::PreprocessingDialect>(
        [&](Operation* op) { return typeConverter.isLegal(op); });
    target
        .addIllegalOp<lwe::RLWEEncryptOp, lwe::RLWEDecryptOp, lwe::RLWEEncodeOp,
                      lwe::RLWEDecodeOp, lwe::RAddOp, lwe::RSubOp, lwe::RMulOp,
                      lwe::RMulPlainOp, lwe::RSubPlainOp, lwe::RAddPlainOp>();

    RewritePatternSet patterns(context);
    addStructuralConversionPatterns(typeConverter, patterns, target);
    addTensorConversionPatterns(typeConverter, patterns, target);
    addMemRefConversionPatterns(typeConverter, patterns, target);
    preprocessing::populatePreprocessingConversions(patterns, typeConverter,
                                                    context);

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      bool hasCryptoContextArg =
          op.getFunctionType().getNumInputs() > 0 &&
          containsArgumentOfType<
              lattigo::BGVEvaluatorType, lattigo::BGVEncoderType,
              lattigo::CKKSEvaluatorType, lattigo::CKKSEncoderType,
              lattigo::RLWEEncryptorType, lattigo::RLWEDecryptorType>(op);
      bool hasCryptoArg =
          containsArgumentOfDialect<lwe::LWEDialect, bgv::BGVDialect,
                                    ckks::CKKSDialect, lattigo::LattigoDialect>(
              op);
      bool hasEncodeOp = containsEncode(op);

      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody()) &&
             (!(hasCryptoArg || hasEncodeOp) || hasCryptoContextArg);
    });

    // Ensures that callee function signature is consistent
    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      auto operandTypes = op.getCalleeType().getInputs();
      auto containsCryptoArg = llvm::any_of(operandTypes, [&](Type argType) {
        return DialectEqual<lwe::LWEDialect, bgv::BGVDialect,
                            ckks::CKKSDialect>()(&argType.getDialect());
      });
      auto hasCryptoContextArg =
          !operandTypes.empty() &&
          mlir::isa<lattigo::BGVEvaluatorType, lattigo::CKKSEvaluatorType>(
              *operandTypes.begin());
      // validate that the callee signature is consistent with the call site
      bool signatureConsistent = false;
      FailureOr<func::FuncOp> callee = getCalledFunction(op);
      if (succeeded(callee)) {
        signatureConsistent =
            callee.value().getFunctionType() == op.getCalleeType();
      }
      return (!containsCryptoArg || hasCryptoContextArg) && signatureConsistent;
    });

    // All other operations are legal if they have no LWE typed operands or
    // results
    target.markUnknownOpDynamicallyLegal(
        [&](Operation* op) { return typeConverter.isLegal(op); });

    OpPredicate containsEncryptUseSk = [&](Operation* op) -> bool {
      if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
        // for declaration, assume its uses are decrypt
        if (funcOp.isDeclaration()) {
          return false;
        }
        return llvm::any_of(funcOp.getArguments(), [&](BlockArgument arg) {
          return mlir::isa<lwe::LWESecretKeyType>(arg.getType()) &&
                 llvm::any_of(arg.getUses(), [&](OpOperand& use) {
                   return mlir::isa<lwe::RLWEEncryptOp>(use.getOwner());
                 });
        });
      }
      return false;
    };

    auto gateByBGVModuleAttr =
        [&](const OpPredicate& inputPredicate) -> OpPredicate {
      return [module, inputPredicate](Operation* op) {
        return moduleIsBGVOrBFV(module) && inputPredicate(op);
      };
    };

    auto gateByCKKSModuleAttr =
        [&](const OpPredicate& inputPredicate) -> OpPredicate {
      return [module, inputPredicate](Operation* op) {
        return moduleIsCKKS(module) && inputPredicate(op);
      };
    };

    OpPredicate containsNoEncryptUseSk = [&](Operation* op) -> bool {
      if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
        bool findKey =
            llvm::any_of(funcOp.getArgumentTypes(), [&](Type argType) {
              return mlir::isa<lwe::LWESecretKeyType>(argType);
            });
        // for declaration, only checks the existence
        if (funcOp.isDeclaration()) {
          return findKey;
        }
        // for definition, check the uses
        bool noEncrypt =
            llvm::all_of(funcOp.getArguments(), [&](BlockArgument arg) {
              return !mlir::isa<lwe::LWESecretKeyType>(arg.getType()) ||
                     llvm::none_of(arg.getUses(), [&](OpOperand& use) {
                       return mlir::isa<lwe::RLWEEncryptOp>(use.getOwner());
                     });
            });
        return findKey && noEncrypt;
      }
      return false;
    };

    std::vector<std::pair<Type, OpPredicate>> evaluators;

    // param/encoder also needed for the main func
    // as there might (not) be ct-pt operations
    evaluators = {
        {lattigo::BGVEvaluatorType::get(context),
         gateByBGVModuleAttr(
             containsArgumentOfDialect<lwe::LWEDialect, bgv::BGVDialect>)},
        {lattigo::BGVParameterType::get(context),
         gateByBGVModuleAttr([&](Operation* op) {
           return containsArgumentOfDialect<lwe::LWEDialect, bgv::BGVDialect>(
                      op) ||
                  containsEncode(op);
         })},
        {lattigo::BGVEncoderType::get(context),
         gateByBGVModuleAttr([&](Operation* op) {
           return containsArgumentOfDialect<lwe::LWEDialect, bgv::BGVDialect>(
                      op) ||
                  containsEncode(op);
         })},
        // Add a CKKS bootstrapping evaluator - this contains a pointer to a
        // CKKS evaluator, and for simplicity, the function signature can
        // contain both. Callers will create only a bootstrapper evaluator and
        // pass the bootstrap_eval.Evaluator in as well.
        {lattigo::CKKSBootstrappingEvaluatorType::get(context),
         gateByCKKSModuleAttr([&](Operation* op) {
           return containsArgumentOfDialect<lwe::LWEDialect, ckks::CKKSDialect>(
                      op) &&
                  containsBootstrap(op);
         })},
        {lattigo::CKKSEvaluatorType::get(context),
         gateByCKKSModuleAttr(
             containsArgumentOfDialect<lwe::LWEDialect, ckks::CKKSDialect>)},
        {lattigo::CKKSParameterType::get(context),
         gateByCKKSModuleAttr([&](Operation* op) {
           return containsArgumentOfDialect<lwe::LWEDialect, ckks::CKKSDialect>(
                      op) ||
                  containsEncode(op);
         })},
        {lattigo::CKKSEncoderType::get(context),
         gateByCKKSModuleAttr([&](Operation* op) {
           return containsArgumentOfDialect<lwe::LWEDialect, ckks::CKKSDialect>(
                      op) ||
                  containsEncode(op);
         })},
        {lattigo::RLWEEncryptorType::get(context, /*publicKey*/ true),
         containsArgumentOfType<lwe::LWEPublicKeyType>},
        // for LWESecretKey, if its uses are encrypt, then convert it to an
        // encryptor, otherwise, convert it to a decryptor
        {lattigo::RLWEEncryptorType::get(context, /*publicKey*/ false),
         containsEncryptUseSk},
        {lattigo::RLWEDecryptorType::get(context), containsNoEncryptUseSk},
    };

    patterns.add<AddEvaluatorArg>(context, evaluators);
    patterns.add<ConvertFuncCallOp>(context, evaluators);

    if (moduleIsBGVOrBFV(module)) {
      patterns.add<ConvertBGVAddOp, ConvertBGVSubOp, ConvertBGVMulOp,
                   ConvertBGVAddPlainOp, ConvertBGVSubPlainOp,
                   ConvertBGVMulPlainOp, ConvertBGVRelinOp,
                   ConvertBGVModulusSwitchOp, ConvertBGVRotateColumnsOp,
                   ConvertBGVEncryptOp, ConvertBGVDecryptOp, ConvertBGVEncodeOp,
                   ConvertBGVDecodeOp, ConvertBGVLevelReduceOp>(typeConverter,
                                                                context);
    }
    if (moduleIsCKKS(module)) {
      patterns.add<
          ConvertCKKSAddOp, ConvertCKKSSubOp, ConvertCKKSMulOp,
          ConvertCKKSAddPlainOp, ConvertCKKSSubPlainOp, ConvertCKKSMulPlainOp,
          ConvertCKKSRelinOp, ConvertCKKSModulusSwitchOp, ConvertCKKSRotateOp,
          ConvertCKKSEncryptOp, ConvertCKKSDecryptOp, ConvertCKKSEncodeOp,
          ConvertCKKSDecodeOp, ConvertCKKSLevelReduceOp, ConvertCKKSBootstrapOp,
          ConvertOrionLinearTransformOp, ConvertOrionChebyshevOp>(typeConverter,
                                                                  context);
    }
    // Misc

    ConversionConfig config;
    config.allowPatternRollback = false;
    if (failed(applyPartialConversion(module, target, std::move(patterns),
                                      config))) {
      return signalPassFailure();
    }

    // remove key args from function calls
    // walkAndApplyPatterns will cause segfault at MLIR side
    RewritePatternSet postPatterns(context);
    postPatterns.add<RemoveKeyArgForFuncCall<lattigo::RLWESecretKeyType>>(
        context);
    postPatterns.add<RemoveKeyArgForFuncCall<lattigo::RLWEPublicKeyType>>(
        context);

    ConversionTarget postTarget(*context);
    postTarget.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      return llvm::none_of(op.getOperandTypes(), [&](Type operandType) {
        return mlir::isa<lattigo::RLWESecretKeyType,
                         lattigo::RLWEPublicKeyType>(operandType);
      });
    });

    ConversionConfig postConfig;
    postConfig.allowPatternRollback = false;
    if (failed(applyPartialConversion(module, postTarget,
                                      std::move(postPatterns), postConfig))) {
      return signalPassFailure();
    }

    // remove unused key args from function types
    // in favor of encryptor/decryptor
    RewritePatternSet postPatterns2(context);
    postPatterns2.add<RemoveKeyArg<lattigo::RLWESecretKeyType>>(context);
    postPatterns2.add<RemoveKeyArg<lattigo::RLWEPublicKeyType>>(context);
    walkAndApplyPatterns(module, std::move(postPatterns2));

    // Restore the dialect attributes of func::CallOp after conversion.
    restoreFuncCallOpDialectAttrs();
  }
};

}  // namespace mlir::heir::lwe
