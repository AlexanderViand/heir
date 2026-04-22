#ifndef LIB_TRANSFORMS_POPULATESCALE_POPULATESCALEPATTERNS_H_
#define LIB_TRANSFORMS_POPULATESCALE_POPULATESCALEPATTERNS_H_

#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "llvm/include/llvm/ADT/APInt.h"        // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"   // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project

namespace mlir {
namespace heir {

class AdjustScaleMaterializer {
 public:
  virtual FailureOr<APInt> deltaScale(mgmt::AdjustScaleOp op,
                                      const APInt& scale,
                                      const APInt& inputScale) const = 0;
};

template <typename MulOp>
struct ConvertAdjustScaleToMulPlain
    : public OpRewritePattern<mgmt::AdjustScaleOp> {
  using OpRewritePattern<mgmt::AdjustScaleOp>::OpRewritePattern;

  ConvertAdjustScaleToMulPlain(MLIRContext* context,
                               AdjustScaleMaterializer* materializer)
      : OpRewritePattern<mgmt::AdjustScaleOp>(context, /*benefit=*/1),
        materializer(materializer) {}

  LogicalResult matchAndRewrite(mgmt::AdjustScaleOp op,
                                PatternRewriter& rewriter) const override;

 private:
  AdjustScaleMaterializer* materializer;
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_POPULATESCALE_POPULATESCALEPATTERNS_H_
