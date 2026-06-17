#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Mgmt/Transforms/AnnotateMgmt.h"
#include "lib/Dialect/Mgmt/Transforms/Passes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
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

    // The Cheddar backend has a fixed canonical scale per level. We read the
    // backend from the module attribute set by annotate-module (run before this
    // pass in the cheddar pipeline) and switch to a scale-management strategy
    // that never emits adjust_scale.
    bool cheddarMode = moduleIsCheddar(getOperation());

    InsertMgmtPipelineOptions options;
    options.includeFloats = true;
    options.levelBudget = levelBudget;
    // Cheddar requires rescale-after-mult: this guarantees every multiplication
    // result immediately drops a level, so every operand-scale mismatch becomes
    // a cross-level mismatch (resolved with level_down, no adjust_scale). With
    // rescale-before-mult, a not-yet-rescaled mul result would instead force a
    // same-level adjust_scale (MatchCrossMulDepth) that Cheddar cannot
    // represent.
    options.modReduceAfterMul = afterMul || cheddarMode;
    options.modReduceBeforeMulIncludeFirstMul = beforeMulIncludeFirstMul;
    options.bootstrapWaterline = bootstrapWaterline;
    // In cheddar mode, cross-level mismatches are resolved with level_reduce
    // (cheddar.level_down) instead of adjust_scale + mod_reduce.
    options.cheddarMode = cheddarMode;
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
    pipeline.addPass(mgmt::createAnnotateMgmt());
    (void)runPipeline(pipeline, getOperation());

    // In cheddar mode, cross-level mismatches are resolved with level_down and
    // no adjust_scale should ever be emitted. adjust_scale lowers to a scalar
    // mul_plain with the nominal scale, which Cheddar rejects, so verify it is
    // absent rather than letting it silently corrupt scales downstream.
    if (cheddarMode) {
      bool foundAdjustScale = false;
      getOperation()->walk([&](mgmt::AdjustScaleOp op) {
        op.emitError(
            "cheddar backend does not support mgmt.adjust_scale; "
            "cross-level mismatches must be resolved with level_down");
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
