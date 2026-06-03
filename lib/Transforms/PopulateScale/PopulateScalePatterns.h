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
  virtual ~AdjustScaleMaterializer() = default;

  virtual FailureOr<APInt> deltaScale(mgmt::AdjustScaleOp op,
                                      const APInt& scale,
                                      const APInt& inputScale) const = 0;

  // Returns the scale of the materialized product (input * delta) in this
  // scheme's scale domain. CKKS scales are linear, so this is the exact
  // (width-extending) product; BGV scales live in Z_t, so it must be reduced
  // mod the plaintext modulus. The shared materialization pattern annotates the
  // resulting mul with this scale, so getting the domain wrong corrupts every
  // downstream consumer of the scale (e.g. encoding scaling factors).
  virtual APInt resultScale(const APInt& inputScale,
                            const APInt& deltaScale) const = 0;
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
