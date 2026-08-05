#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Mgmt/Transforms/AnnotateMgmt.h"
#include "lib/Dialect/Mgmt/Transforms/Passes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Target/CompilationTarget/CompilationTarget.h"
#include "lib/Transforms/SecretInsertMgmt/Passes.h"
#include "lib/Transforms/SecretInsertMgmt/Pipeline.h"
#include "llvm/include/llvm/Support/Debug.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"         // from @llvm-project

#define DEBUG_TYPE "secret-insert-mgmt-ckks"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_SECRETINSERTMGMTCKKS
#include "lib/Transforms/SecretInsertMgmt/Passes.h.inc"

struct SecretInsertMgmtCKKS
    : impl::SecretInsertMgmtCKKSBase<SecretInsertMgmtCKKS> {
  using SecretInsertMgmtCKKSBase::SecretInsertMgmtCKKSBase;

  void runOnOperation() override {
    // Helper for future lowerings that want to know what scheme was used
    moduleSetCKKS(getOperation());

    // Backends with one fixed canonical scale per level (cheddar) cannot
    // represent mgmt.adjust_scale; read the capability from the backend's
    // CompilationTarget entry (annotate-module ran earlier and stamped the
    // backend attribute the registry resolves from).
    bool noAdjustScale = false;
    if (auto target = getTargetConfig(cast<ModuleOp>(getOperation()));
        succeeded(target))
      noAdjustScale = !target->can_emit_adjust_scale;

    InsertMgmtPipelineOptions options;
    options.includeFloats = true;
    options.levelBudget = levelBudget;
    // Such backends require rescale-after-mult: this guarantees every
    // multiplication result immediately drops a level, so every operand-scale
    // mismatch becomes a cross-level mismatch (resolved with level_reduce, no
    // adjust_scale). With rescale-before-mult, a not-yet-rescaled mul result
    // would instead force a same-level adjust_scale (MatchCrossMulDepth) the
    // backend cannot represent.
    options.modReduceAfterMul = afterMul || noAdjustScale;
    options.modReduceBeforeMulIncludeFirstMul = beforeMulIncludeFirstMul;
    options.bootstrapWaterline = bootstrapWaterline;
    // Without adjust_scale, cross-level mismatches are resolved with
    // level_reduce alone.
    options.noAdjustScale = noAdjustScale;
    LogicalResult result = runInsertMgmtPipeline(getOperation(), options);

    if (failed(result)) {
      signalPassFailure();
      return;
    }

    LLVM_DEBUG(llvm::dbgs() << "Post secret-insert-mgmt pipeline cleanup\n");

    // 1. Canonicalizer reorders mgmt ops like Rescale/LevelReduce/AdjustScale.
    //    This is important for AnnotateMgmt.
    //    Canonicalizer also moves mgmt::InitOp out of secret.generic.
    // 2. CSE removes redundant mgmt::ModReduceOp.
    // 3. Canonicalizer will remove mgmt.level_reduce_min ops since now the
    //    level information is concrete.
    // 4. AnnotateMgmt will merge level and dimension into MgmtAttr, for further
    //   lowering.
    OpPassManager pipeline("builtin.module");
    pipeline.addPass(createCanonicalizerPass());
    pipeline.addPass(createCSEPass());
    mgmt::AnnotateMgmtOptions annotateOptions;
    annotateOptions.levelBudget = levelBudget;
    pipeline.addPass(mgmt::createAnnotateMgmt(annotateOptions));
    (void)runPipeline(pipeline, getOperation());

    // When the target cannot represent adjust_scale, none should ever be
    // emitted; verify that rather than letting one silently corrupt scales
    // downstream (on cheddar it would lower to a scalar mul_plain at a
    // nominal scale the backend rejects).
    if (noAdjustScale) {
      bool foundAdjustScale = false;
      getOperation()->walk([&](mgmt::AdjustScaleOp op) {
        op.emitError(
            "the target backend does not support mgmt.adjust_scale; "
            "cross-level mismatches must be resolved with level_reduce");
        foundAdjustScale = true;
      });
      if (foundAdjustScale) {
        signalPassFailure();
        return;
      }
    }
  }
};

}  // namespace heir
}  // namespace mlir
