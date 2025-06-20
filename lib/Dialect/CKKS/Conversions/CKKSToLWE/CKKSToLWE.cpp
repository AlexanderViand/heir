#include "lib/Dialect/CKKS/Conversions/CKKSToLWE/CKKSToLWE.h"

#include <utility>

#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWEPatterns.h"
#include "lib/Utils/RewriteUtils/RewriteUtils.h"
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

namespace mlir::heir::ckks {

#define GEN_PASS_DEF_CKKSTOLWE
#include "lib/Dialect/CKKS/Conversions/CKKSToLWE/CKKSToLWE.h.inc"

struct CKKSToLWE : public impl::CKKSToLWEBase<CKKSToLWE> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();

    RewritePatternSet patterns(context);
    patterns
        .add<Convert<AddOp, lwe::RAddOp>, Convert<AddPlainOp, lwe::RAddPlainOp>,
             Convert<SubOp, lwe::RSubOp>, Convert<SubPlainOp, lwe::RSubPlainOp>,
             Convert<NegateOp, lwe::RNegateOp>, Convert<MulOp, lwe::RMulOp>,
             Convert<MulPlainOp, lwe::RMulPlainOp>>(context);
    walkAndApplyPatterns(module, std::move(patterns));
  }
};

}  // namespace mlir::heir::ckks
