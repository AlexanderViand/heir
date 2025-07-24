#include "lib/Transforms/AddMemrefAlignment/AddMemrefAlignment.h"

#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                 // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_ADDMEMREFALIGNMENTPASS
#include "lib/Transforms/AddMemrefAlignment/AddMemrefAlignment.h.inc"

namespace {

// Pattern to add alignment attributes to memref.alloc operations
class AddAlignmentToAllocOp : public OpRewritePattern<memref::AllocOp> {
 public:
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp op,
                                PatternRewriter &rewriter) const override {
    // If the alloc operation already has an alignment attribute, don't modify
    // it
    if (op.getAlignmentAttr()) {
      return failure();
    }

    // Create a new AllocOp with 64-byte alignment
    auto alignmentAttr = rewriter.getI64IntegerAttr(64);

    // Create the new operation with alignment
    SmallVector<NamedAttribute> newAttrs;
    for (auto attr : op->getAttrs()) {
      newAttrs.push_back(attr);
    }
    newAttrs.push_back(
        NamedAttribute(rewriter.getStringAttr("alignment"), alignmentAttr));

    auto newAllocOp = rewriter.create<memref::AllocOp>(
        op.getLoc(), op.getType(), op.getDynamicSizes(), newAttrs);

    rewriter.replaceOp(op, newAllocOp.getResult());
    return success();
  }
};

struct AddMemrefAlignmentPass
    : impl::AddMemrefAlignmentPassBase<AddMemrefAlignmentPass> {
  using AddMemrefAlignmentPassBase::AddMemrefAlignmentPassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<AddAlignmentToAllocOp>(context);

    // Apply the patterns greedily
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace heir
}  // namespace mlir
