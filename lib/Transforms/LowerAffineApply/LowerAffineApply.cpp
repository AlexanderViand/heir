#include "lib/Transforms/LowerAffineApply/LowerAffineApply.h"

#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Utils.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"             // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_LOWERAFFINEAPPLY
#include "lib/Transforms/LowerAffineApply/LowerAffineApply.h.inc"

namespace {

struct LowerAffineApply : impl::LowerAffineApplyBase<LowerAffineApply> {
  using LowerAffineApplyBase::LowerAffineApplyBase;

  void runOnOperation() override {
    // Collect first: expandAffineMap inserts new ops and we erase each
    // affine.apply, so we avoid mutating while walking.
    SmallVector<affine::AffineApplyOp> applyOps;
    getOperation()->walk(
        [&](affine::AffineApplyOp op) { applyOps.push_back(op); });

    IRRewriter rewriter(&getContext());
    for (affine::AffineApplyOp op : applyOps) {
      rewriter.setInsertionPoint(op);
      std::optional<SmallVector<Value, 8>> expanded =
          affine::expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(),
                                  llvm::to_vector(op.getMapOperands()));
      if (!expanded) {
        op.emitOpError("failed to expand affine map to arithmetic ops");
        return signalPassFailure();
      }
      // affine.apply always has a single-result affine map.
      rewriter.replaceOp(op, expanded->front());
    }
  }
};

}  // namespace

}  // namespace heir
}  // namespace mlir
