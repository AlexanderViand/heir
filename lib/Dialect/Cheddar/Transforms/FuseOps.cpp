#include "lib/Dialect/Cheddar/Transforms/FuseOps.h"

#include <cstdint>
#include <optional>

#include "lib/Dialect/Cheddar/IR/CheddarDialect.h"
#include "lib/Dialect/Cheddar/IR/CheddarOps.h"
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StaticValueUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"         // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"        // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir::heir::cheddar {

namespace {

struct FuseMultRelinRescale : public OpRewritePattern<RescaleOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RescaleOp rescaleOp,
                                PatternRewriter& rewriter) const override {
    auto relinOp = rescaleOp.getInput().getDefiningOp<RelinearizeOp>();
    if (!relinOp || !relinOp->getResult(0).hasOneUse()) return failure();
    auto multOp = relinOp.getInput().getDefiningOp<MultOp>();
    if (!multOp || !multOp->getResult(0).hasOneUse()) return failure();
    if (multOp.getCtx() != relinOp.getCtx() ||
        multOp.getCtx() != rescaleOp.getCtx())
      return failure();

    auto fused = HMultOp::create(
        rewriter, rescaleOp.getLoc(), rescaleOp->getResultTypes(),
        multOp.getCtx(), multOp.getLhs(), multOp.getRhs(), relinOp.getMultKey(),
        rescaleOp.getOutput(), rewriter.getBoolAttr(true));
    rewriter.replaceOp(rescaleOp, fused);
    rewriter.eraseOp(relinOp);
    rewriter.eraseOp(multOp);
    return success();
  }
};

struct FuseMultRelin : public OpRewritePattern<RelinearizeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RelinearizeOp relinOp,
                                PatternRewriter& rewriter) const override {
    if (relinOp->getResult(0).hasOneUse() &&
        isa<RescaleOp>(*relinOp->getResult(0).getUsers().begin()))
      return failure();
    auto multOp = relinOp.getInput().getDefiningOp<MultOp>();
    if (!multOp || !multOp->getResult(0).hasOneUse() ||
        multOp.getCtx() != relinOp.getCtx())
      return failure();

    auto fused = HMultOp::create(
        rewriter, relinOp.getLoc(), relinOp->getResultTypes(), multOp.getCtx(),
        multOp.getLhs(), multOp.getRhs(), relinOp.getMultKey(),
        relinOp.getOutput(), rewriter.getBoolAttr(false));
    rewriter.replaceOp(relinOp, fused);
    rewriter.eraseOp(multOp);
    return success();
  }
};

struct FuseMultRelinRescaleFused
    : public OpRewritePattern<RelinearizeRescaleOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RelinearizeRescaleOp relinRescaleOp,
                                PatternRewriter& rewriter) const override {
    auto multOp = relinRescaleOp.getInput().getDefiningOp<MultOp>();
    if (!multOp || !multOp->getResult(0).hasOneUse() ||
        multOp.getCtx() != relinRescaleOp.getCtx())
      return failure();

    auto fused = HMultOp::create(
        rewriter, relinRescaleOp.getLoc(), relinRescaleOp->getResultTypes(),
        multOp.getCtx(), multOp.getLhs(), multOp.getRhs(),
        relinRescaleOp.getMultKey(), relinRescaleOp.getOutput(),
        rewriter.getBoolAttr(true));
    rewriter.replaceOp(relinRescaleOp, fused);
    rewriter.eraseOp(multOp);
    return success();
  }
};

template <typename OpTy>
OpTy fusableAddend(AddOp addOp, Value& otherOperand) {
  if (auto lhs = addOp.getLhs().getDefiningOp<OpTy>();
      lhs && lhs->getResult(0).hasOneUse()) {
    otherOperand = addOp.getRhs();
    return lhs;
  }
  if (auto rhs = addOp.getRhs().getDefiningOp<OpTy>();
      rhs && rhs->getResult(0).hasOneUse()) {
    otherOperand = addOp.getLhs();
    return rhs;
  }
  return nullptr;
}

struct FuseHRotAdd : public OpRewritePattern<AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp addOp,
                                PatternRewriter& rewriter) const override {
    Value otherOperand;
    HRotOp hrotOp = fusableAddend<HRotOp>(addOp, otherOperand);
    if (!hrotOp || hrotOp.getCtx() != addOp.getCtx()) return failure();

    IntegerAttr distanceAttr;
    if (auto staticDistance = hrotOp.getStaticDistanceAttr()) {
      distanceAttr = staticDistance;
    } else if (Value dynamicDistance = hrotOp.getDynamicDistance()) {
      std::optional<int64_t> constantDistance =
          getConstantIntValue(dynamicDistance);
      if (!constantDistance) return failure();
      distanceAttr = rewriter.getI64IntegerAttr(*constantDistance);
    } else {
      return failure();
    }

    auto fused = HRotAddOp::create(
        rewriter, addOp.getLoc(), addOp->getResultTypes(), hrotOp.getCtx(),
        hrotOp.getEvk(), hrotOp.getInput(), otherOperand, addOp.getOutput(),
        distanceAttr, hrotOp.getLevelAttr());
    rewriter.replaceOp(addOp, fused);
    rewriter.eraseOp(hrotOp);
    return success();
  }
};

struct FuseHConjAdd : public OpRewritePattern<AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp addOp,
                                PatternRewriter& rewriter) const override {
    Value otherOperand;
    HConjOp hconjOp = fusableAddend<HConjOp>(addOp, otherOperand);
    if (!hconjOp || hconjOp.getCtx() != addOp.getCtx()) return failure();

    auto fused = HConjAddOp::create(
        rewriter, addOp.getLoc(), addOp->getResultTypes(), hconjOp.getCtx(),
        hconjOp.getEvk(), hconjOp.getInput(), otherOperand, addOp.getOutput());
    rewriter.replaceOp(addOp, fused);
    rewriter.eraseOp(hconjOp);
    return success();
  }
};

template <typename PlainOp>
struct HoistRelinBeforePlainOp : public OpRewritePattern<RelinearizeOp> {
  using OpRewritePattern<RelinearizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(RelinearizeOp relinOp,
                                PatternRewriter& rewriter) const override {
    auto plainOp = relinOp.getInput().template getDefiningOp<PlainOp>();
    if (!plainOp || !plainOp->getResult(0).hasOneUse()) return failure();
    auto multOp = plainOp.getCiphertext().template getDefiningOp<MultOp>();
    if (!multOp || !multOp->getResult(0).hasOneUse()) return failure();
    if (multOp.getCtx() != plainOp.getCtx() ||
        multOp.getCtx() != relinOp.getCtx())
      return failure();

    rewriter.setInsertionPointAfter(multOp);
    auto tensorType = cast<RankedTensorType>(multOp->getResult(0).getType());
    Value newRelinDest = tensor::EmptyOp::create(rewriter, relinOp.getLoc(),
                                                 tensorType.getShape(),
                                                 tensorType.getElementType());
    auto newRelin = RelinearizeOp::create(
        rewriter, relinOp.getLoc(), relinOp->getResultTypes(), multOp.getCtx(),
        multOp->getResult(0), relinOp.getMultKey(), newRelinDest);

    rewriter.setInsertionPoint(relinOp);
    auto newPlain = PlainOp::create(
        rewriter, relinOp.getLoc(), relinOp->getResultTypes(), plainOp.getCtx(),
        newRelin->getResult(0), plainOp.getPlaintext(), relinOp.getOutput());
    rewriter.replaceOp(relinOp, newPlain);
    rewriter.eraseOp(plainOp);
    return success();
  }
};

}  // namespace

#define GEN_PASS_DEF_CHEDDARFUSEOPS
#include "lib/Dialect/Cheddar/Transforms/FuseOps.h.inc"

namespace {

struct CheddarFuseOps : public impl::CheddarFuseOpsBase<CheddarFuseOps> {
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<HoistRelinBeforePlainOp<AddPlainOp>>(context, /*benefit=*/4);
    patterns.add<HoistRelinBeforePlainOp<SubPlainOp>>(context, /*benefit=*/4);
    patterns.add<FuseMultRelinRescale>(context, /*benefit=*/3);
    patterns.add<FuseMultRelinRescaleFused>(context, /*benefit=*/2);
    patterns.add<FuseMultRelin>(context, /*benefit=*/1);
    patterns.add<FuseHRotAdd, FuseHConjAdd>(context, /*benefit=*/1);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

}  // namespace

}  // namespace mlir::heir::cheddar
