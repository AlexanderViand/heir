#include "include/Transforms/ElementwiseToAffine/ElementwiseToAffine.h"

#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm - project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"  // from @llvm-project
namespace mlir {
namespace heir {

#define GEN_PASS_DEF_ELEMENTWISETOAFFINE
#include "include/Transforms/ElementwiseToAffine/ElementwiseToAffine.h.inc"

// All of this is based on the ElementwiseToLinalg Pass in
// mlir/lib/Dialect/Linalg/Transforms/ElementwiseToLinalg.cpp

static bool isElementwiseMappableOpOnRankedTensors(Operation *op) {
  if (!OpTrait::hasElementwiseMappableTraits(op)) return false;

  // TODO: The conversion pattern can be made to work for `any_of` here, but
  // it's more complex as it requires tracking which operands are scalars.
  return llvm::all_of(op->getOperandTypes(),
                      [](Type type) { return isa<RankedTensorType>(type); });
}

/// Given `op` assumed `isElementwiseMappableOpOnRankedTensors`, iterate over
/// the result types and return a list of values such that, for each result type
/// `t` and value `v` at the same index `idx`:
///   1. `v.getType() == t`
///   2. If an operand of `op` has type `t`, let `operand_first` be the first
///      such operand. Then`v == operand_first`.
///   3. Otherwise, v is a newly created `tensor::EmptyOp` with:
///        a. Static and dynamic dims extracted from the first operand of `op`.
///        b. Elemental type equal to the elemental type of `t`.
///
/// This is sufficient because ElementwiseMappable guarantees that "The static
/// types of all vector (resp. tensor) operands and results must have the same
/// shape".
static SmallVector<Value, 4> getOrCreateOperandsMatchingResultTypes(
    OpBuilder &b, Operation *op) {
  assert(isElementwiseMappableOpOnRankedTensors(op));
  Location loc = op->getLoc();
  ValueRange operands = op->getOperands();
  TypeRange rankedTensorTypes = op->getResultTypes();
  SmallVector<Value, 4> res;
  res.reserve(rankedTensorTypes.size());
  for (Type t : rankedTensorTypes) {
    // Try to find an operand with type matching the result tensor.
    bool found = false;
    for (Value v : operands) {
      if (v.getType() == t) {
        found = true;
        res.push_back(v);
        break;
      }
    }
    if (found) continue;

    // Extract static / dynamic shape mix from the first operand.
    res.push_back(b.create<tensor::EmptyOp>(
        loc, tensor::getMixedSizes(b, loc, operands.front()),
        cast<RankedTensorType>(t).getElementType()));
  }
  return res;
}

namespace {

struct ConvertAnyElementwiseMappableOpOnRankedTensors : public RewritePattern {
  ConvertAnyElementwiseMappableOpOnRankedTensors(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final {
    if (!isElementwiseMappableOpOnRankedTensors(op))
      return rewriter.notifyMatchFailure(
          op, "requires elementwise op on ranked tensors");

    auto rank = cast<RankedTensorType>(op->getResult(0).getType()).getRank();
    SmallVector<AffineMap, 3> indexingMaps(
        op->getNumResults() + op->getNumOperands(),
        rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType, 6> iteratorTypes(
        rank, utils::IteratorType::parallel);
    auto outputs = getOrCreateOperandsMatchingResultTypes(rewriter, op);
    rewriter.replaceOpWithNewOp<linalg::GenericOp>(  // FIXME: use affine.for
        op, /*resultTensorTypes=*/op->getResultTypes(),
        /*inputs=*/op->getOperands(),
        /*outputs=*/outputs,
        /*indexingMaps=*/indexingMaps,
        /*iteratorTypes=*/iteratorTypes,
        /*bodyBuilder=*/
        [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
          auto resultTypes = llvm::to_vector<6>(
              llvm::map_range(op->getResultTypes(), [](Type type) {
                return cast<TensorType>(type).getElementType();
              }));
          // FIXME: Generate a `tensor.extract` for each operand

          // FIXME: Replace the operands here with the extract operands
          auto *scalarOp =
              builder.create(loc, op->getName().getIdentifier(),
                             regionArgs.take_front(op->getNumOperands()),
                             resultTypes, op->getAttrs());

          // FIXME: insert scalarOp into the tensor at the right index

          // FIXME: replace lingalg.yield scalarOp with affine.yield insertedOp
          builder.create<linalg::YieldOp>(loc, scalarOp->getResults());
          // builder.create<affine::AffineYieldOp>(loc, scalarOp->getResults());
        });
    return success();
  }
};
}  // namespace

struct ElementwiseToAffine
    : impl::ElementwiseToAffineBase<ElementwiseToAffine> {
  using ElementwiseToAffineBase::ElementwiseToAffineBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    RewritePatternSet patterns(context);

    patterns.add<ConvertAnyElementwiseMappableOpOnRankedTensors>(context);
    target.markUnknownOpDynamicallyLegal([](Operation *op) {
      return !isElementwiseMappableOpOnRankedTensors(op);
    });

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

}  // namespace heir
}  // namespace mlir
