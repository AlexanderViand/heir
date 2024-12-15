#include "lib/Dialect/CKKS/Conversions/CKKSToLWE/CKKSToLWE.h"

#include <utility>

#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWEPatterns.h"
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir::heir::ckks {

#define GEN_PASS_DEF_CKKSTOLWE
#include "lib/Dialect/CKKS/Conversions/CKKSToLWE/CKKSToLWE.h.inc"

template <typename CKKSOp, typename LWEOp>
struct Convert : public OpRewritePattern<CKKSOp> {
  Convert(mlir::MLIRContext *context) : OpRewritePattern<CKKSOp>(context) {}

  LogicalResult matchAndRewrite(CKKSOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LWEOp>(op, op->getOperands(), op->getAttrs());
    return success();
  }
};

struct CKKSToLWE : public impl::CKKSToLWEBase<CKKSToLWE> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();

    RewritePatternSet patterns(context);
    patterns.add<Convert<AddOp, lwe::RAddOp>, Convert<SubOp, lwe::RSubOp>,
                 Convert<NegateOp, lwe::RNegateOp>, Convert<MulOp, lwe::RMulOp>,
                 lwe::ConvertExtract<ExtractOp, MulPlainOp, RotateOp> >(
        context);
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace mlir::heir::ckks
