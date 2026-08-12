#include "lib/Dialect/Cheddar/Transforms/PrepareLinearTransforms.h"

#include <string>
#include <utility>

#include "lib/Dialect/Cheddar/IR/CheddarOps.h"
#include "lib/Dialect/Cheddar/IR/CheddarTypes.h"
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/SetVector.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "mlir/include/mlir/Analysis/SliceAnalysis.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/IRMapping.h"              // from @llvm-project
#include "mlir/include/mlir/IR/SymbolTable.h"            // from @llvm-project

namespace mlir::heir::cheddar {

#define GEN_PASS_DEF_CHEDDARPREPARELINEARTRANSFORMS
#include "lib/Dialect/Cheddar/Transforms/PrepareLinearTransforms.h.inc"

namespace {

static FailureOr<std::pair<func::CallOp, func::CallOp>> findCallPair(
    ModuleOp module, func::FuncOp preprocessing, func::FuncOp preprocessed) {
  func::CallOp foundPack;
  func::CallOp foundEval;
  unsigned evalCallCount = 0;
  module.walk([&](func::CallOp call) {
    if (call.getCallee() != preprocessed.getName()) return;
    ++evalCallCount;
    for (Value operand : call.getOperands()) {
      auto pack = operand.getDefiningOp<func::CallOp>();
      if (pack && pack.getCallee() == preprocessing.getName()) {
        foundPack = pack;
        foundEval = call;
        return;
      }
    }
  });
  if (evalCallCount != 1 || !foundPack || !foundEval) return failure();
  return std::make_pair(foundPack, foundEval);
}

static LogicalResult appendFunctionResult(func::FuncOp func, Type type,
                                          Value value) {
  if (!llvm::hasSingleElement(func.getBody()) ||
      !isa<func::ReturnOp>(func.getBody().front().getTerminator()))
    return func.emitOpError(
        "linear-transform preprocessing requires one block and one return");
  SmallVector<Type> results(func.getResultTypes());
  results.push_back(type);
  func.setFunctionType(
      FunctionType::get(func.getContext(), func.getArgumentTypes(), results));
  auto ret = cast<func::ReturnOp>(func.getBody().front().getTerminator());
  ret.getOperandsMutable().append(value);
  return success();
}

static FailureOr<Value> cloneDiagonalSlice(LinearTransformOp raw,
                                           OpBuilder& builder,
                                           IRMapping& mapper) {
  if (Value mapped = mapper.lookupOrNull(raw.getDiagonals())) return mapped;
  Operation* root = raw.getDiagonals().getDefiningOp();
  if (!root)
    return raw.emitOpError(
        "cleartext diagonals must be produced by cloneable operations");
  llvm::SetVector<Operation*> slice;
  BackwardSliceOptions options;
  options.omitBlockArguments = true;
  if (failed(getBackwardSlice(root, &slice, options)))
    return raw.emitOpError("failed to compute cleartext diagonal slice");

  auto clone = [&](Operation* operation) -> LogicalResult {
    if (mapper.contains(operation)) return success();
    for (Value operand : operation->getOperands()) {
      if (!mapper.contains(operand))
        return raw.emitOpError(
            "cleartext diagonal construction captures a value unavailable "
            "in preprocessing");
    }
    Operation* cloned = builder.clone(*operation, mapper);
    mapper.map(operation, cloned);
    return success();
  };
  for (Operation* dependency : slice)
    if (failed(clone(dependency))) return failure();
  if (failed(clone(root))) return failure();
  Value diagonals = mapper.lookupOrNull(raw.getDiagonals());
  if (!diagonals)
    return raw.emitOpError("could not clone cleartext diagonal construction");
  return diagonals;
}

struct CheddarPrepareLinearTransforms
    : impl::CheddarPrepareLinearTransformsBase<CheddarPrepareLinearTransforms> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<func::FuncOp> evalFuncs;
    module.walk([&](func::FuncOp func) {
      if (func.getName().ends_with("__preprocessed") &&
          !func.getOps<LinearTransformOp>().empty())
        evalFuncs.push_back(func);
    });

    for (func::FuncOp evalFunc : evalFuncs) {
      std::string base = evalFunc.getName()
                             .drop_back(StringRef("__preprocessed").size())
                             .str();
      auto packFunc =
          module.lookupSymbol<func::FuncOp>(base + "__preprocessing");
      if (!packFunc) {
        evalFunc.emitOpError("has linear transforms but no paired ")
            << base << "__preprocessing function";
        return signalPassFailure();
      }
      auto calls = findCallPair(module, packFunc, evalFunc);
      if (failed(calls)) {
        evalFunc.emitOpError(
            "requires exactly one directly paired preprocessing wrapper call");
        return signalPassFailure();
      }
      if (!llvm::hasSingleElement(packFunc.getBody()) ||
          !llvm::hasSingleElement(evalFunc.getBody())) {
        evalFunc.emitOpError(
            "linear-transform preprocessing requires single-block functions");
        return signalPassFailure();
      }
      func::CallOp oldPackCall = calls->first;
      func::CallOp oldEvalCall = calls->second;

      SmallVector<LinearTransformOp> rawTransforms(
          evalFunc.getOps<LinearTransformOp>());
      Value rawCtx = rawTransforms.front().getCtx();
      auto ctxArg = dyn_cast<BlockArgument>(rawCtx);
      if (!ctxArg || ctxArg.getOwner() != &evalFunc.getBody().front()) {
        rawTransforms.front().emitOpError(
            "expected linear-transform context to be a function argument");
        return signalPassFailure();
      }
      Value wrapperCtx = oldEvalCall.getOperand(ctxArg.getArgNumber());
      if (failed(packFunc.insertArgument(packFunc.getNumArguments(),
                                         rawCtx.getType(), nullptr,
                                         packFunc.getLoc()))) {
        packFunc.emitOpError("failed to append context argument");
        return signalPassFailure();
      }
      BlockArgument packCtx = packFunc.getArguments().back();

      IRMapping mapper;
      mapper.map(rawCtx, packCtx);
      for (auto [evalIndex, wrapperOperand] :
           llvm::enumerate(oldEvalCall.getOperands())) {
        Value evalArg = evalFunc.getArgument(evalIndex);
        for (auto [packIndex, packOperand] :
             llvm::enumerate(oldPackCall.getOperands()))
          if (wrapperOperand == packOperand)
            mapper.map(evalArg, packFunc.getArgument(packIndex));
      }

      OpBuilder packBuilder(packFunc.getBody().front().getTerminator());
      OpBuilder evalBuilder(evalFunc.getContext());
      auto handleType = RankedTensorType::get(
          {}, LinearTransformType::get(evalFunc.getContext()));
      unsigned oldPackResultCount = oldPackCall.getNumResults();
      unsigned preparedCount = 0;

      for (LinearTransformOp raw : rawTransforms) {
        FailureOr<Value> clonedDiagonals =
            cloneDiagonalSlice(raw, packBuilder, mapper);
        if (failed(clonedDiagonals)) return signalPassFailure();

        auto empty = tensor::EmptyOp::create(packBuilder, raw.getLoc(),
                                             ArrayRef<int64_t>{},
                                             handleType.getElementType());
        auto prepared = PrepareLinearTransformOp::create(
            packBuilder, raw.getLoc(), handleType, packCtx, *clonedDiagonals,
            empty, raw.getDiagonalIndicesAttr(),
            packBuilder.getI64IntegerAttr(
                cast<ShapedType>(raw.getDiagonals().getType()).getDimSize(1)),
            raw.getLevelAttr(), raw.getBsAttr(), raw.getGsAttr(),
            raw.getMinKsAttr());
        if (failed(appendFunctionResult(packFunc, handleType,
                                        prepared.getResult().front())))
          return signalPassFailure();

        if (failed(evalFunc.insertArgument(evalFunc.getNumArguments(),
                                           handleType, nullptr,
                                           raw.getLoc()))) {
          raw.emitOpError("failed to append prepared-transform argument");
          return signalPassFailure();
        }
        BlockArgument handleArg = evalFunc.getArguments().back();
        evalBuilder.setInsertionPoint(raw);
        auto apply = ApplyPreparedLinearTransformOp::create(
            evalBuilder, raw.getLoc(), raw.getResultTypes(), raw.getCtx(),
            raw.getInput(), raw.getEvkMap(), handleArg, raw.getOutput(),
            raw.getMinKsAttr());
        raw->replaceAllUsesWith(apply);
        raw.erase();
        ++preparedCount;
      }

      OpBuilder callBuilder(oldPackCall);
      SmallVector<Value> packOperands(oldPackCall.getOperands());
      packOperands.push_back(wrapperCtx);
      auto newPackCall = func::CallOp::create(
          callBuilder, oldPackCall.getLoc(), packFunc.getName(),
          packFunc.getResultTypes(), packOperands);
      for (auto [oldResult, newResult] :
           llvm::zip(oldPackCall.getResults(), newPackCall.getResults()))
        oldResult.replaceAllUsesWith(newResult);

      callBuilder.setInsertionPoint(oldEvalCall);
      SmallVector<Value> evalOperands(oldEvalCall.getOperands());
      for (unsigned i = 0; i < preparedCount; ++i)
        evalOperands.push_back(newPackCall.getResult(oldPackResultCount + i));
      auto newEvalCall = func::CallOp::create(
          callBuilder, oldEvalCall.getLoc(), evalFunc.getName(),
          evalFunc.getResultTypes(), evalOperands);
      oldEvalCall.replaceAllUsesWith(newEvalCall.getResults());
      oldEvalCall.erase();
      oldPackCall.erase();
    }

    // Constant linear transforms have no encoded plaintext for the generic
    // SplitPreprocessing analysis to discover. If no split pair exists, create
    // one now and keep the original public function as the wrapper. This is
    // the common native-model path (e.g. frozen matvec weights).
    SmallVector<func::FuncOp> unsplitFuncs;
    module.walk([&](func::FuncOp func) {
      if (!func.getName().contains("__") &&
          !func.getOps<LinearTransformOp>().empty())
        unsplitFuncs.push_back(func);
    });
    for (func::FuncOp wrapper : unsplitFuncs) {
      SmallVector<LinearTransformOp> rawTransforms(
          wrapper.getOps<LinearTransformOp>());
      if (rawTransforms.empty()) continue;
      if (!llvm::hasSingleElement(wrapper.getBody())) {
        wrapper.emitOpError(
            "linear-transform preprocessing requires a single-block function");
        return signalPassFailure();
      }
      Value rawCtx = rawTransforms.front().getCtx();
      auto ctxArg = dyn_cast<BlockArgument>(rawCtx);
      if (!ctxArg || ctxArg.getOwner() != &wrapper.getBody().front()) {
        rawTransforms.front().emitOpError(
            "expected linear-transform context to be a function argument");
        return signalPassFailure();
      }

      std::string packName = wrapper.getName().str() + "__preprocessing";
      std::string evalName = wrapper.getName().str() + "__preprocessed";
      if (module.lookupSymbol(packName) || module.lookupSymbol(evalName)) {
        wrapper.emitOpError("cannot create linear-transform helpers because '")
            << packName << "' or '" << evalName << "' already exists";
        return signalPassFailure();
      }
      OpBuilder moduleBuilder(wrapper);
      SmallVector<Type> packInputs{rawCtx.getType()};
      SmallVector<Type> handleTypes(
          rawTransforms.size(),
          RankedTensorType::get(
              {}, LinearTransformType::get(wrapper.getContext())));
      auto pack = func::FuncOp::create(
          moduleBuilder, wrapper.getLoc(), packName,
          FunctionType::get(wrapper.getContext(), packInputs, handleTypes));
      pack.setVisibility(wrapper.getVisibility());
      Block* packBlock = pack.addEntryBlock();
      OpBuilder packBuilder = OpBuilder::atBlockEnd(packBlock);

      auto eval = cast<func::FuncOp>(moduleBuilder.clone(*wrapper));
      eval.setName(evalName);
      SmallVector<BlockArgument> handleArgs;
      for (Type handleType : handleTypes) {
        if (failed(eval.insertArgument(eval.getNumArguments(), handleType,
                                       nullptr, eval.getLoc()))) {
          eval.emitOpError("failed to append prepared-transform argument");
          return signalPassFailure();
        }
        handleArgs.push_back(eval.getArguments().back());
      }

      IRMapping mapper;
      mapper.map(rawCtx, packBlock->getArgument(0));
      SmallVector<Value> preparedValues;
      unsigned transformIndex = 0;
      for (auto raw :
           llvm::make_early_inc_range(eval.getOps<LinearTransformOp>())) {
        FailureOr<Value> diagonals =
            cloneDiagonalSlice(raw, packBuilder, mapper);
        if (failed(diagonals)) return signalPassFailure();
        auto empty = tensor::EmptyOp::create(
            packBuilder, raw.getLoc(), ArrayRef<int64_t>{},
            cast<RankedTensorType>(handleTypes[transformIndex])
                .getElementType());
        auto prepared = PrepareLinearTransformOp::create(
            packBuilder, raw.getLoc(), handleTypes[transformIndex],
            packBlock->getArgument(0), *diagonals, empty,
            raw.getDiagonalIndicesAttr(),
            packBuilder.getI64IntegerAttr(
                cast<ShapedType>(raw.getDiagonals().getType()).getDimSize(1)),
            raw.getLevelAttr(), raw.getBsAttr(), raw.getGsAttr(),
            raw.getMinKsAttr());
        preparedValues.push_back(prepared.getResult().front());

        OpBuilder evalBuilder(raw);
        auto apply = ApplyPreparedLinearTransformOp::create(
            evalBuilder, raw.getLoc(), raw.getResultTypes(), raw.getCtx(),
            raw.getInput(), raw.getEvkMap(), handleArgs[transformIndex],
            raw.getOutput(), raw.getMinKsAttr());
        raw->replaceAllUsesWith(apply);
        raw.erase();
        ++transformIndex;
      }
      func::ReturnOp::create(packBuilder, pack.getLoc(), preparedValues);

      Block& wrapperBlock = wrapper.getBody().front();
      wrapper.getBody().dropAllReferences();
      wrapperBlock.clear();
      OpBuilder wrapperBuilder = OpBuilder::atBlockEnd(&wrapperBlock);
      auto packed = func::CallOp::create(
          wrapperBuilder, wrapper.getLoc(), pack.getName(), handleTypes,
          ValueRange{wrapper.getArgument(ctxArg.getArgNumber())});
      SmallVector<Value> evalOperands(wrapper.getArguments());
      llvm::append_range(evalOperands, packed.getResults());
      auto evaluated =
          func::CallOp::create(wrapperBuilder, wrapper.getLoc(), eval.getName(),
                               eval.getResultTypes(), evalOperands);
      func::ReturnOp::create(wrapperBuilder, wrapper.getLoc(),
                             evaluated.getResults());
    }
  }
};

}  // namespace
}  // namespace mlir::heir::cheddar
