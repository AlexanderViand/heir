#include "lib/Dialect/Cheddar/Transforms/FuseOps.h"

#include <cstdint>
#include <optional>

#include "lib/Dialect/Cheddar/IR/CheddarDialect.h"
#include "lib/Dialect/Cheddar/IR/CheddarOps.h"
#include "lib/Dialect/Cheddar/IR/CheddarTypes.h"
#include "mlir/include/mlir/Dialect/Bufferization/IR/Bufferization.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StaticValueUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"         // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"        // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir::heir::cheddar {

//===----------------------------------------------------------------------===//
// Fusion patterns
//===----------------------------------------------------------------------===//
//
// These run at the cheddar tensor (pre-bufferization, destination-passing)
// level. A fusion that *replaces* an op reuses that op's `$output` operand as
// the fused op's destination; a fusion that inserts a *new* intermediate op
// allocates a fresh `bufferization.alloc_tensor` destination for it.

// Pattern: mult + relinearize + rescale -> hmult(rescale=true)
struct FuseMultRelinRescale : public OpRewritePattern<RescaleOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RescaleOp rescaleOp,
                                PatternRewriter &rewriter) const override {
    // Check that the input to rescale is a relinearize
    auto relinOp = rescaleOp.getInput().getDefiningOp<RelinearizeOp>();
    if (!relinOp || !relinOp->getResult(0).hasOneUse()) return failure();

    // Check that the input to relinearize is a mult
    auto multOp = relinOp.getInput().getDefiningOp<MultOp>();
    if (!multOp || !multOp->getResult(0).hasOneUse()) return failure();

    // Fuse into HMult with rescale=true, writing into rescale's destination.
    rewriter.replaceOpWithNewOp<HMultOp>(
        rescaleOp, rescaleOp->getResultTypes(), multOp.getCtx(),
        multOp.getLhs(), multOp.getRhs(), relinOp.getMultKey(),
        rescaleOp.getOutput(),
        /*rescale=*/rewriter.getBoolAttr(true));

    // Clean up now-dead ops
    rewriter.eraseOp(relinOp);
    rewriter.eraseOp(multOp);
    return success();
  }
};

// Pattern: mult + relinearize -> hmult(rescale=false)
struct FuseMultRelin : public OpRewritePattern<RelinearizeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RelinearizeOp relinOp,
                                PatternRewriter &rewriter) const override {
    // Don't match if this relin feeds into a rescale (handled by
    // FuseMultRelinRescale).
    if (relinOp->getResult(0).hasOneUse()) {
      auto *user = *relinOp->getResult(0).getUsers().begin();
      if (isa<RescaleOp>(user)) return failure();
    }

    auto multOp = relinOp.getInput().getDefiningOp<MultOp>();
    if (!multOp || !multOp->getResult(0).hasOneUse()) return failure();

    rewriter.replaceOpWithNewOp<HMultOp>(
        relinOp, relinOp->getResultTypes(), multOp.getCtx(), multOp.getLhs(),
        multOp.getRhs(), relinOp.getMultKey(), relinOp.getOutput(),
        /*rescale=*/rewriter.getBoolAttr(false));

    rewriter.eraseOp(multOp);
    return success();
  }
};

// Pattern: mult + relinearize_rescale -> hmult(rescale=true)
struct FuseMultRelinRescaleFused
    : public OpRewritePattern<RelinearizeRescaleOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RelinearizeRescaleOp relinRescaleOp,
                                PatternRewriter &rewriter) const override {
    auto multOp = relinRescaleOp.getInput().getDefiningOp<MultOp>();
    if (!multOp || !multOp->getResult(0).hasOneUse()) return failure();

    rewriter.replaceOpWithNewOp<HMultOp>(
        relinRescaleOp, relinRescaleOp->getResultTypes(), multOp.getCtx(),
        multOp.getLhs(), multOp.getRhs(), relinRescaleOp.getMultKey(),
        relinRescaleOp.getOutput(),
        /*rescale=*/rewriter.getBoolAttr(true));

    rewriter.eraseOp(multOp);
    return success();
  }
};

// Find a single-use `OpTy` defining one of `addOp`'s operands; on success
// `otherOperand` is set to the other addend. Shared by the hrot/hconj fusions.
template <typename OpTy>
OpTy fusableAddend(AddOp addOp, Value &otherOperand) {
  if (auto lhs = addOp.getLhs().getDefiningOp<OpTy>())
    if (lhs->getResult(0).hasOneUse()) {
      otherOperand = addOp.getRhs();
      return lhs;
    }
  if (auto rhs = addOp.getRhs().getDefiningOp<OpTy>())
    if (rhs->getResult(0).hasOneUse()) {
      otherOperand = addOp.getLhs();
      return rhs;
    }
  return nullptr;
}

// Pattern: hrot(a) + b -> hrot_add(a, b)
// `hrot_add` carries a static distance attribute, so it can fuse an hrot with a
// static distance or one whose dynamic distance is a constant (the form the
// rotate-and-sum lowering produces, e.g. `cheddar.hrot %ctx, %a, %c5`).
struct FuseHRotAdd : public OpRewritePattern<AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp addOp,
                                PatternRewriter &rewriter) const override {
    Value otherOperand;
    HRotOp hrotOp = fusableAddend<HRotOp>(addOp, otherOperand);
    if (!hrotOp) return failure();

    // Resolve the rotation distance to the static attribute hrot_add carries.
    IntegerAttr distanceAttr;
    if (auto staticDist = hrotOp.getStaticDistanceAttr()) {
      distanceAttr = staticDist;
    } else if (Value dyn = hrotOp.getDynamicDistance()) {
      std::optional<int64_t> c = getConstantIntValue(dyn);
      if (!c) return failure();  // non-constant dynamic distance: cannot fuse
      distanceAttr = rewriter.getI64IntegerAttr(*c);
    } else {
      return failure();
    }

    rewriter.replaceOpWithNewOp<HRotAddOp>(
        addOp, addOp->getResultTypes(), hrotOp.getCtx(), hrotOp.getInput(),
        otherOperand, addOp.getOutput(), distanceAttr);
    rewriter.eraseOp(hrotOp);
    return success();
  }
};

// Pattern: hconj(a) + b -> hconj_add(a, b)
struct FuseHConjAdd : public OpRewritePattern<AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp addOp,
                                PatternRewriter &rewriter) const override {
    Value otherOperand;
    HConjOp hconjOp = fusableAddend<HConjOp>(addOp, otherOperand);
    if (!hconjOp) return failure();

    rewriter.replaceOpWithNewOp<HConjAddOp>(
        addOp, addOp->getResultTypes(), hconjOp.getCtx(), hconjOp.getInput(),
        otherOperand, addOp.getOutput());
    rewriter.eraseOp(hconjOp);
    return success();
  }
};

// Pattern: move relinearize before add_plain/sub_plain when the plain op's
// ciphertext is an unrelinearized mult result (a degree-3 ciphertext). CHEDDAR
// doesn't support add/sub on degree-3 ciphertexts, but the CKKS pipeline may
// schedule {add,sub}_plain between mult and relin (valid for Lattigo/OpenFHE).
// relin and {add,sub}_plain commute (they touch independent ciphertext
// components), so reorder:
//   %a = mult(%x,%y); %b = {add,sub}_plain(%a,%pt); %c = relin(%b,%key)
//   => %a = mult(%x,%y); %a2 = relin(%a,%key); %c = {add,sub}_plain(%a2,%pt)
// This also re-exposes the mult+relin -> hmult fusion above.
template <typename PlainOp>
struct HoistRelinBeforePlainOp : public OpRewritePattern<RelinearizeOp> {
  using OpRewritePattern<RelinearizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(RelinearizeOp relinOp,
                                PatternRewriter &rewriter) const override {
    auto plainOp = relinOp.getInput().template getDefiningOp<PlainOp>();
    if (!plainOp || !plainOp->getResult(0).hasOneUse()) return failure();

    auto multOp = plainOp.getCiphertext().template getDefiningOp<MultOp>();
    if (!multOp || !multOp->getResult(0).hasOneUse()) return failure();

    // Insert relinearize right after the mult (before the plain op). The new
    // relin is a fresh intermediate, so it needs its own destination buffer.
    rewriter.setInsertionPointAfter(multOp);
    auto tensorTy = cast<RankedTensorType>(multOp->getResult(0).getType());
    Value newRelinDest = bufferization::AllocTensorOp::create(
        rewriter, relinOp.getLoc(), tensorTy, ValueRange{});
    auto newRelin = RelinearizeOp::create(
        rewriter, relinOp.getLoc(), relinOp->getResultTypes(), multOp.getCtx(),
        multOp->getResult(0), relinOp.getMultKey(), newRelinDest);

    // Replace the plain op's ciphertext input with the relinearized result;
    // the fused plain op writes into the original relin's destination. Build it
    // at relinOp's position (not the moved insertion point above) so that
    // destination -- defined just before relinOp -- dominates the new op.
    rewriter.setInsertionPoint(relinOp);
    rewriter.replaceOpWithNewOp<PlainOp>(
        relinOp, relinOp->getResultTypes(), plainOp.getCtx(),
        newRelin->getResult(0), plainOp.getPlaintext(), relinOp.getOutput());
    rewriter.eraseOp(plainOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

#define GEN_PASS_DEF_CHEDDARFUSEOPS
#include "lib/Dialect/Cheddar/Transforms/FuseOps.h.inc"

struct CheddarFuseOps : public impl::CheddarFuseOpsBase<CheddarFuseOps> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // First hoist relinearize before add/sub_plain so CHEDDAR never sees a
    // degree-3 ciphertext in an add/sub, and the mult+relin fusions below can
    // match (higher benefit so it runs before the fusions).
    patterns.add<HoistRelinBeforePlainOp<AddPlainOp>>(context, /*benefit=*/4);
    patterns.add<HoistRelinBeforePlainOp<SubPlainOp>>(context, /*benefit=*/4);

    // Order matters: try the longest fusions first.
    patterns.add<FuseMultRelinRescale>(context, /*benefit=*/3);
    patterns.add<FuseMultRelinRescaleFused>(context, /*benefit=*/2);
    patterns.add<FuseMultRelin>(context, /*benefit=*/1);
    patterns.add<FuseHRotAdd>(context, /*benefit=*/1);
    patterns.add<FuseHConjAdd>(context, /*benefit=*/1);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace mlir::heir::cheddar
