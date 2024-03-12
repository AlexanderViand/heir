#include "include/Transforms/ElementwiseToAffine/ElementwiseToAffine.h"

#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_ELEMENTWISETOAFFINE
#include "include/Transforms/ElementwiseToAffine/ElementwiseToAffine.h.inc"

struct ElementwiseToAffine
    : impl::ElementwiseToAffineBase<ElementwiseToAffine> {
  using ElementwiseToAffineBase::ElementwiseToAffineBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // FIXME: implement pass
    // patterns.add<>(context);

    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
