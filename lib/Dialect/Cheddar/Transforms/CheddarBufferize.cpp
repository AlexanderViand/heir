#include "lib/Dialect/Cheddar/Transforms/CheddarBufferize.h"

#include <optional>
#include <utility>

#include "lib/Dialect/Cheddar/IR/CheddarDialect.h"
#include "lib/Dialect/Cheddar/IR/CheddarOps.h"
#include "lib/Dialect/Cheddar/IR/CheddarTypes.h"
#include "llvm/include/llvm/ADT/BitVector.h"  // from @llvm-project
#include "llvm/include/llvm/ADT/DenseMap.h"   // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Bufferization/IR/Bufferization.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Bufferization/Transforms/Transforms.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"            // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar {

#define GEN_PASS_DEF_CHEDDARBUFFERIZE
#include "lib/Dialect/Cheddar/Transforms/Passes.h.inc"

namespace {

bool isCheddarResultTensor(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  return tensorType &&
         tensorType.getElementType().getDialect().getNamespace() ==
             CheddarDialect::getDialectNamespace();
}

// Materialize packed Cheddar results element-by-element. A tensor.insert_slice
// has value-copy semantics and therefore always asks One-Shot Bufferize for a
// memcpy. At a function return the destination buffer is already explicit, so
// decompose the packing chain into materializations of the inserted values
// directly into rank-reduced subviews of that buffer. This preserves the
// source tensor semantics while exposing the actual destination before
// analysis.
LogicalResult materializeResultInDestination(
    OpBuilder& builder, Value value, Value destination,
    SmallVectorImpl<Operation*>& deadPackingOps) {
  auto insert = value.getDefiningOp<tensor::InsertSliceOp>();
  if (insert && isCheddarResultTensor(insert.getSource().getType()) &&
      insert.getResult().hasOneUse()) {
    if (failed(materializeResultInDestination(builder, insert.getDest(),
                                              destination, deadPackingOps)))
      return failure();

    auto destinationType = cast<MemRefType>(destination.getType());
    auto sourceType = insert.getSourceType();
    auto subviewType = memref::SubViewOp::inferRankReducedResultType(
        sourceType.getShape(), destinationType, insert.getMixedOffsets(),
        insert.getMixedSizes(), insert.getMixedStrides());
    Value subview = memref::SubViewOp::create(
                        builder, insert.getLoc(), subviewType, destination,
                        insert.getMixedOffsets(), insert.getMixedSizes(),
                        insert.getMixedStrides())
                        .getResult();
    if (failed(materializeResultInDestination(builder, insert.getSource(),
                                              subview, deadPackingOps)))
      return failure();
    deadPackingOps.push_back(insert);
    return success();
  }

  // A loop-carried tensor result can use the caller-provided buffer directly.
  // Materialize the loop's initial value in that buffer, expose it as a
  // writable tensor, and make it the corresponding iter_arg. Subset insertion
  // inside the loop can then extract its DPS destination from the iter_arg;
  // stock One-Shot Bufferize keeps the loop result equivalent to the caller's
  // buffer without an aggregate copy.
  if (auto result = dyn_cast<OpResult>(value)) {
    if (auto forOp = dyn_cast<scf::ForOp>(result.getOwner())) {
      unsigned resultNumber = result.getResultNumber();
      if (resultNumber >= forOp.getInitArgs().size())
        return forOp.emitOpError("result has no corresponding init argument");

      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(forOp);
      Value init = forOp.getInitArgs()[resultNumber];
      if (failed(materializeResultInDestination(builder, init, destination,
                                                deadPackingOps)))
        return failure();
      auto tensor = bufferization::ToTensorOp::create(
          builder, forOp.getLoc(), cast<RankedTensorType>(result.getType()),
          destination, builder.getUnitAttr(), builder.getUnitAttr());
      forOp.getInitArgsMutable()[resultNumber].set(tensor.getResult());
      return success();
    }
  }

  if (auto empty = value.getDefiningOp<tensor::EmptyOp>()) {
    if (empty.getResult().hasOneUse()) deadPackingOps.push_back(empty);
    return success();
  }

  bufferization::MaterializeInDestinationOp::create(
      builder, value.getLoc(), TypeRange{}, value, destination,
      builder.getUnitAttr(), builder.getUnitAttr());
  return success();
}

std::optional<unsigned> findTiedFunctionArgument(func::FuncOp function,
                                                 Value value) {
  while (true) {
    for (auto [index, argument] : llvm::enumerate(function.getArguments())) {
      if (value == argument) return index;
    }
    auto result = dyn_cast<OpResult>(value);
    if (!result) return std::nullopt;
    auto dpsOp = dyn_cast<DestinationStyleOpInterface>(result.getOwner());
    if (!dpsOp) return std::nullopt;
    OpOperand* tiedInit = dpsOp.getTiedOpOperand(result);
    if (!tiedInit) return std::nullopt;
    value = tiedInit->get();
  }
}

std::optional<unsigned> getEquivalentArgumentForResult(func::FuncOp function,
                                                       unsigned resultIndex) {
  std::optional<unsigned> equivalentArgument;
  bool sawReturn = false;
  WalkResult walkResult = function.walk([&](func::ReturnOp returnOp) {
    sawReturn = true;
    if (resultIndex >= returnOp.getNumOperands())
      return WalkResult::interrupt();
    std::optional<unsigned> argument =
        findTiedFunctionArgument(function, returnOp.getOperand(resultIndex));
    if (!argument || (equivalentArgument && equivalentArgument != argument))
      return WalkResult::interrupt();
    equivalentArgument = argument;
    return WalkResult::advance();
  });
  if (!sawReturn || walkResult.wasInterrupted()) return std::nullopt;
  return equivalentArgument;
}

// Make caller-provided tensor result storage visible while the program still
// has tensor semantics. One-Shot can then propagate each destination backward
// through DPS and subset-insertion chains instead of returning a memref (and,
// after EmitC conversion, an unsupported C++ array value).
LogicalResult materializeTensorResultsInDestinations(ModuleOp module) {
  struct FunctionConversion {
    func::FuncOp function;
    SmallVector<unsigned> resultIndices;
    SmallVector<RankedTensorType> resultTypes;
    SmallVector<int64_t> equivalentArgumentByResult;
  };

  SmallVector<FunctionConversion> conversions;
  DenseMap<StringAttr, unsigned> conversionByName;
  module.walk([&](func::FuncOp function) {
    if (function.isExternal()) return;
    FunctionConversion conversion{
        function,
        {},
        {},
        SmallVector<int64_t>(function.getNumResults(), /*Value=*/-1)};
    for (auto [index, type] : llvm::enumerate(function.getResultTypes())) {
      if (!isa<RankedTensorType>(type)) continue;
      if (std::optional<unsigned> argument =
              getEquivalentArgumentForResult(function, index)) {
        conversion.equivalentArgumentByResult[index] = *argument;
        continue;
      }
      conversion.resultIndices.push_back(index);
      conversion.resultTypes.push_back(cast<RankedTensorType>(type));
    }
    bool hasEquivalentResult =
        llvm::any_of(conversion.equivalentArgumentByResult,
                     [](int64_t argument) { return argument >= 0; });
    if (conversion.resultIndices.empty() && !hasEquivalentResult) return;
    conversionByName[function.getSymNameAttr()] = conversions.size();
    conversions.push_back(std::move(conversion));
  });

  for (FunctionConversion& conversion : conversions) {
    func::FuncOp function = conversion.function;
    SmallVector<BlockArgument> outArgs;
    outArgs.reserve(conversion.resultIndices.size());
    for (auto [resultIndex, tensorType] :
         llvm::zip(conversion.resultIndices, conversion.resultTypes)) {
      NamedAttrList attrs(function.getResultAttrDict(resultIndex));
      attrs.set("bufferize.result", UnitAttr::get(module.getContext()));
      auto memrefType =
          MemRefType::get(tensorType.getShape(), tensorType.getElementType());
      if (failed(function.insertArgument(
              function.getNumArguments(), memrefType,
              attrs.getDictionary(function.getContext()), function.getLoc())))
        return function.emitOpError("failed to append result destination");
      outArgs.push_back(function.getArguments().back());
    }

    SmallVector<func::ReturnOp> returns;
    function.walk(
        [&](func::ReturnOp returnOp) { returns.push_back(returnOp); });
    for (func::ReturnOp returnOp : returns) {
      if (returnOp.getNumOperands() != function.getNumResults())
        return returnOp.emitOpError(
            "return operand count does not match function results");
      OpBuilder builder(returnOp);
      SmallVector<Value> keptOperands;
      SmallVector<Operation*> deadPackingOps;
      unsigned convertedIndex = 0;
      for (auto [resultIndex, value] :
           llvm::enumerate(returnOp.getOperands())) {
        if (conversion.equivalentArgumentByResult[resultIndex] >= 0) continue;
        if (convertedIndex < conversion.resultIndices.size() &&
            conversion.resultIndices[convertedIndex] == resultIndex) {
          if (failed(materializeResultInDestination(
                  builder, value, outArgs[convertedIndex], deadPackingOps)))
            return failure();
          ++convertedIndex;
        } else {
          keptOperands.push_back(value);
        }
      }
      returnOp.getOperandsMutable().assign(keptOperands);
      for (Operation* op : llvm::reverse(deadPackingOps)) {
        if (op->use_empty()) op->erase();
      }
    }

    BitVector resultsToErase(function.getNumResults());
    for (unsigned index : conversion.resultIndices) resultsToErase.set(index);
    for (auto [index, argument] :
         llvm::enumerate(conversion.equivalentArgumentByResult))
      if (argument >= 0) resultsToErase.set(index);
    if (failed(function.eraseResults(resultsToErase)))
      return function.emitOpError("failed to erase materialized results");
  }

  SmallVector<func::CallOp> calls;
  module.walk([&](func::CallOp call) {
    if (conversionByName.contains(
            StringAttr::get(module.getContext(), call.getCallee())))
      calls.push_back(call);
  });

  for (func::CallOp call : calls) {
    StringAttr calleeName =
        StringAttr::get(module.getContext(), call.getCallee());
    FunctionConversion& conversion =
        conversions[conversionByName.lookup(calleeName)];
    OpBuilder builder(call);
    SmallVector<Value> newOperands(call.getOperands());
    SmallVector<Type> keptResultTypes;
    SmallVector<std::pair<Value, bufferization::MaterializeInDestinationOp>>
        destinations;
    destinations.reserve(conversion.resultIndices.size());

    unsigned convertedIndex = 0;
    for (auto [resultIndex, result] : llvm::enumerate(call.getResults())) {
      if (conversion.equivalentArgumentByResult[resultIndex] >= 0) continue;
      if (convertedIndex >= conversion.resultIndices.size() ||
          conversion.resultIndices[convertedIndex] != resultIndex) {
        keptResultTypes.push_back(result.getType());
        continue;
      }

      bufferization::MaterializeInDestinationOp forwarding;
      Value destination;
      if (result.hasOneUse()) {
        forwarding = dyn_cast<bufferization::MaterializeInDestinationOp>(
            *result.getUsers().begin());
        if (forwarding && forwarding.getSource() == result) {
          destination = forwarding.getDest();
          if (auto toTensor =
                  destination.getDefiningOp<bufferization::ToTensorOp>())
            destination = toTensor.getBuffer();
          if (!isa<MemRefType>(destination.getType())) forwarding = nullptr;
        } else {
          forwarding = nullptr;
        }
      }
      if (forwarding) {
        // `destination` was recovered from the materialization above. In the
        // usual function-result path it is the buffer wrapped by to_tensor.
      } else {
        RankedTensorType tensorType = conversion.resultTypes[convertedIndex];
        if (!tensorType.hasStaticShape())
          return call.emitOpError(
              "dynamic Cheddar result destinations are not yet supported");
        auto memrefType =
            MemRefType::get(tensorType.getShape(), tensorType.getElementType());
        destination =
            memref::AllocOp::create(builder, call.getLoc(), memrefType)
                .getResult();
      }
      newOperands.push_back(destination);
      destinations.push_back({destination, forwarding});
      ++convertedIndex;
    }

    auto newCall = func::CallOp::create(
        builder, call.getLoc(), call.getCallee(), keptResultTypes, newOperands);
    newCall->setAttrs(call->getAttrDictionary());

    builder.setInsertionPointAfter(newCall);
    convertedIndex = 0;
    unsigned keptIndex = 0;
    for (auto [resultIndex, result] : llvm::enumerate(call.getResults())) {
      int64_t equivalentArgument =
          conversion.equivalentArgumentByResult[resultIndex];
      if (equivalentArgument >= 0) {
        result.replaceAllUsesWith(call.getOperand(equivalentArgument));
        continue;
      }
      if (convertedIndex < conversion.resultIndices.size() &&
          conversion.resultIndices[convertedIndex] == resultIndex) {
        auto [destination, forwarding] = destinations[convertedIndex];
        if (forwarding) {
          forwarding.erase();
        } else {
          auto tensor = bufferization::ToTensorOp::create(
              builder, call.getLoc(), conversion.resultTypes[convertedIndex],
              destination, builder.getUnitAttr(), builder.getUnitAttr());
          result.replaceAllUsesWith(tensor.getResult());
        }
        ++convertedIndex;
      } else {
        result.replaceAllUsesWith(newCall.getResult(keptIndex++));
      }
    }
    call.erase();
  }

  return success();
}

struct CheddarBufferize : public impl::CheddarBufferizeBase<CheddarBufferize> {
  using Base::Base;

  void getDependentDialects(DialectRegistry& registry) const override {
    registry
        .insert<bufferization::BufferizationDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(materializeTensorResultsInDestinations(module))) {
      signalPassFailure();
      return;
    }

    IRRewriter rewriter(&getContext());
    if (failed(bufferization::eliminateEmptyTensors(rewriter, module))) {
      signalPassFailure();
      return;
    }

    bufferization::OneShotBufferizationOptions options;
    options.allowReturnAllocsFromLoops = true;
    options.bufferizeFunctionBoundaries = true;
    options.setFunctionBoundaryTypeConversion(
        bufferization::LayoutMapOption::IdentityLayoutMap);
    options.memCpyFn = [](OpBuilder& builder, Location loc, Value source,
                          Value target) -> LogicalResult {
      if (source == target) return success();
      auto sourceType = dyn_cast<BaseMemRefType>(source.getType());
      if (sourceType && isa<UserInterfaceType>(sourceType.getElementType()))
        return emitError(loc)
               << "bufferization cannot copy a Cheddar user interface";
      memref::CopyOp::create(builder, loc, source, target);
      return success();
    };
    bufferization::BufferizationState state;
    if (failed(
            bufferization::runOneShotModuleBufferize(module, options, state))) {
      signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace cheddar
}  // namespace heir
}  // namespace mlir
