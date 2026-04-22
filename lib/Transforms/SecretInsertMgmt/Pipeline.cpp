#include "lib/Transforms/SecretInsertMgmt/Pipeline.h"

#include <utility>

#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/MulDepthAnalysis/MulDepthAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Transforms/Halo/Patterns.h"
#include "lib/Transforms/SecretInsertMgmt/SecretInsertMgmtPatterns.h"
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "llvm/include/llvm/Support/DebugLog.h"            // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/SymbolTable.h"            // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "secret-insert-mgmt"

namespace mlir {
namespace heir {

void runSolver(Operation* top, DataFlowSolver& solver) {
  if (failed(solver.initializeAndRun(top))) {
    LDBG() << "Failed to run solver!";
  }
}

void makeAndRunSolver(Operation* top, DataFlowSolver& solver) {
  dataflow::loadBaselineAnalyses(solver);
  solver.load<SecretnessAnalysis>();
  solver.load<LevelAnalysis>();
  solver.load<MulDepthAnalysis>();
  runSolver(top, solver);
}

void makeAndRunSecretnessSolver(Operation* top, DataFlowSolver& solver) {
  dataflow::loadBaselineAnalyses(solver);
  solver.load<SecretnessAnalysis>();
  runSolver(top, solver);
}

void makeAndRunSecretnessAndMulDepthSolver(Operation* top,
                                           DataFlowSolver& solver) {
  dataflow::loadBaselineAnalyses(solver);
  solver.load<SecretnessAnalysis>();
  solver.load<MulDepthAnalysis>();
  runSolver(top, solver);
}

void makeAndRunSecretnessAndLevelSolver(Operation* top,
                                        DataFlowSolver& solver) {
  dataflow::loadBaselineAnalyses(solver);
  solver.load<SecretnessAnalysis>();
  solver.load<LevelAnalysis>();
  runSolver(top, solver);
}

LogicalResult runInsertMgmtPipeline(Operation* top,
                                    const InsertMgmtPipelineOptions& options) {
  LDBG() << "Starting insert-mgmt pipeline";
  makeLoopsTypeAndLevelInvariant(top);
  LLVM_DEBUG(top->dump());

  insertMgmtInitForPlaintexts(top, options.includeFloats);
  LLVM_DEBUG(top->dump());

  LDBG() << "Inserting mod reduce";
  insertModReduceBeforeOrAfterMult(top, options.modReduceAfterMul,
                                   options.modReduceBeforeMulIncludeFirstMul,
                                   options.includeFloats);
  LLVM_DEBUG(top->dump());

  // this must be run after ModReduceAfterMult
  LDBG() << "Inserting relinearize";
  insertRelinearizeAfterMult(top, options.includeFloats);

  LDBG() << "Unrolling loops for level consumption";
  unrollLoopsForLevelUtilization(top, options.levelBudget);
  LLVM_DEBUG(top->dump());

  // insert BootstrapOp after mgmt::ModReduceOp
  // This must be run before level mismatch
  // NOTE: actually bootstrap before mod reduce is better
  // as after modreduce to level `0` there still might be add/sub
  // and these op done there could be minimal cost.
  // However, this greedy strategy is temporary so not too much
  // optimization now
  if (options.bootstrapWaterline.has_value()) {
    LDBG() << "Bootstrap waterline";
    insertBootstrapWaterLine(top, options.bootstrapWaterline.value());
  }

  // An if statement must have each branch producing the same level as a result,
  // so the branch with the higher level must insert a level_reduce op.
  adjustLevelsForRegionBranchOps(top);

  int idCounter = 0;  // for making adjust_scale op different to avoid cse
  if (options.deferReconcile) {
    LDBG() << "Handling cross level mul ops";
    handleCrossLevelOps(top, &idCounter, options.includeFloats,
                        /*includeAddSub=*/false);
    LDBG() << "Handling cross mul depth mul ops";
    handleCrossMulDepthOps(top, &idCounter, options.includeFloats,
                           /*includeAddSub=*/false);

    // Cross-level rewrites can change yielded ciphertext levels inside region
    // branches, so re-run branch equalization before deferring add/sub repair.
    adjustLevelsForRegionBranchOps(top);

    LDBG() << "Inserting reconcile markers";
    insertReconcileMarkers(top, options.includeFloats);
  } else {
    LDBG() << "Handling cross level ops";
    handleCrossLevelOps(top, &idCounter, options.includeFloats);

    LDBG() << "Handling cross mul depth ops";
    handleCrossMulDepthOps(top, &idCounter, options.includeFloats);

    // Cross-level rewrites can change yielded ciphertext levels inside region
    // branches, so re-run branch equalization after management repair.
    adjustLevelsForRegionBranchOps(top);
  }
  return success();
}

void insertMgmtInitForPlaintexts(Operation* top, bool includeFloats) {
  LDBG() << "Inserting mgmt.init";
  DataFlowSolver solver;
  makeAndRunSecretnessSolver(top, solver);

  MLIRContext* ctx = top->getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<UseInitOpForPlaintextOperand<arith::AddIOp>,
               UseInitOpForPlaintextOperand<arith::SubIOp>,
               UseInitOpForPlaintextOperand<arith::MulIOp>,
               UseInitOpForPlaintextOperand<tensor::ExtractSliceOp>,
               UseInitOpForPlaintextOperand<tensor::InsertSliceOp>>(ctx, top,
                                                                    &solver);

  if (includeFloats) {
    patterns.add<UseInitOpForPlaintextOperand<arith::AddFOp>,
                 UseInitOpForPlaintextOperand<arith::SubFOp>,
                 UseInitOpForPlaintextOperand<arith::MulFOp>>(ctx, top,
                                                              &solver);
  }

  (void)walkAndApplyPatterns(top, std::move(patterns));
}

void insertModReduceBeforeOrAfterMult(Operation* top, bool afterMul,
                                      bool beforeMulIncludeFirstMul,
                                      bool includeFloats) {
  DataFlowSolver solver;
  makeAndRunSecretnessAndMulDepthSolver(top, solver);

  MLIRContext* ctx = top->getContext();
  LLVM_DEBUG({
    auto when = "before mul";
    if (afterMul) when = "after mul";
    if (beforeMulIncludeFirstMul) when = "before mul + before first mul";
    llvm::dbgs() << "Insert ModReduce " << when << "\n";
  });

  RewritePatternSet patterns(ctx);
  if (afterMul) {
    patterns.add<ModReduceAfterMult<arith::MulIOp>>(ctx, top, &solver);
    if (includeFloats)
      patterns.add<ModReduceAfterMult<arith::MulFOp>>(ctx, top, &solver);
  } else {
    patterns.add<ModReduceBefore<arith::MulIOp>>(ctx, beforeMulIncludeFirstMul,
                                                 top, &solver);
    if (includeFloats)
      patterns.add<ModReduceBefore<arith::MulFOp>>(
          ctx, beforeMulIncludeFirstMul, top, &solver);
    // includeFirstMul = false here
    // as before yield we only want mulResult to be mod reduced
    patterns.add<ModReduceBefore<secret::YieldOp>>(
        ctx, /*includeFirstMul*/ false, top, &solver);
    // When before-first-mul is enabled, also insert modreduce before
    // structured ops (diagonal_matvec) to keep noiseScaleDeg bounded.
    // Without this, mul → linear_transform chains accumulate noiseScaleDeg
    // past the CRT limb count in OpenFHE's FIXEDMANUAL mode.
    if (includeFloats && beforeMulIncludeFirstMul)
      patterns.add<ModReduceBefore<tensor_ext::DiagonalMatvecOp>>(
          ctx, /*includeFirstMul*/ false, top, &solver);
  }
  (void)walkAndApplyPatterns(top, std::move(patterns));
}

void insertRelinearizeAfterMult(Operation* top, bool includeFloats) {
  DataFlowSolver solver;
  makeAndRunSecretnessSolver(top, solver);

  MLIRContext* ctx = top->getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<MultRelinearize<arith::MulIOp>>(ctx, top, &solver);
  if (includeFloats)
    patterns.add<MultRelinearize<arith::MulFOp>>(ctx, top, &solver);
  (void)walkAndApplyPatterns(top, std::move(patterns));
}

void handleCrossLevelOps(Operation* top, int* idCounter, bool includeFloats,
                         bool includeAddSub) {
  DataFlowSolver solver;
  makeAndRunSecretnessAndLevelSolver(top, solver);
  MLIRContext* ctx = top->getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<MatchCrossLevel<arith::MulIOp>>(ctx, idCounter, top, &solver);
  if (includeAddSub) {
    patterns
        .add<MatchCrossLevel<arith::AddIOp>, MatchCrossLevel<arith::SubIOp>>(
            ctx, idCounter, top, &solver);
  }
  if (includeFloats) {
    patterns.add<MatchCrossLevel<arith::MulFOp>>(ctx, idCounter, top, &solver);
    if (includeAddSub) {
      patterns
          .add<MatchCrossLevel<arith::AddFOp>, MatchCrossLevel<arith::SubFOp>>(
              ctx, idCounter, top, &solver);
    }
  }
  (void)walkAndApplyPatterns(top, std::move(patterns));
}

// this only happen for before-mul but not include-first-mul case
// at the first level, a Value can be both mulResult or not mulResult
// we should match their scale by adding one adjust scale op
void handleCrossMulDepthOps(Operation* top, int* idCounter, bool includeFloats,
                            bool includeAddSub) {
  DataFlowSolver solver;
  makeAndRunSolver(top, solver);
  MLIRContext* ctx = top->getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<MatchCrossMulDepth<arith::MulIOp>>(ctx, idCounter, top, &solver);
  if (includeAddSub) {
    patterns.add<MatchCrossMulDepth<arith::AddIOp>,
                 MatchCrossMulDepth<arith::SubIOp>>(ctx, idCounter, top,
                                                    &solver);
  }
  if (includeFloats) {
    patterns.add<MatchCrossMulDepth<arith::MulFOp>>(ctx, idCounter, top,
                                                    &solver);
    if (includeAddSub) {
      patterns.add<MatchCrossMulDepth<arith::AddFOp>,
                   MatchCrossMulDepth<arith::SubFOp>>(ctx, idCounter, top,
                                                      &solver);
    }
  }
  (void)walkAndApplyPatterns(top, std::move(patterns));
}

void insertReconcileMarkers(Operation* top, bool includeFloats) {
  DataFlowSolver solver;
  makeAndRunSecretnessSolver(top, solver);
  MLIRContext* ctx = top->getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<InsertReconcileMarker<arith::AddIOp>,
               InsertReconcileMarker<arith::SubIOp>>(ctx, top, &solver);
  if (includeFloats) {
    patterns.add<InsertReconcileMarker<arith::AddFOp>,
                 InsertReconcileMarker<arith::SubFOp>>(ctx, top, &solver);
  }
  (void)walkAndApplyPatterns(top, std::move(patterns));
}

void insertBootstrapWaterLine(Operation* top, int bootstrapWaterline) {
  DataFlowSolver solver;
  makeAndRunSolver(top, solver);
  MLIRContext* ctx = top->getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<BootstrapWaterLine<mgmt::ModReduceOp>>(ctx, top, &solver,
                                                      bootstrapWaterline);
  (void)walkAndApplyPatterns(top, std::move(patterns));
}

void makeLoopsTypeAndLevelInvariant(Operation* top) {
  LDBG() << "Making loops type and level invariant";
  MLIRContext* ctx = top->getContext();

  DataFlowSolver solver;
  makeAndRunSecretnessSolver(top, solver);
  RewritePatternSet patterns(ctx);
  patterns.add<PeelPlaintextAffineForInit, PeelPlaintextScfForInit>(ctx,
                                                                    &solver);
  walkAndApplyPatterns(top, std::move(patterns));

  DataFlowSolver solver2;
  makeAndRunSecretnessSolver(top, solver2);
  patterns.clear();
  patterns.add<BootstrapIterArgsPattern<affine::AffineForOp>,
               BootstrapIterArgsPattern<scf::ForOp>>(ctx, &solver2);
  walkAndApplyPatterns(top, std::move(patterns));

  DataFlowSolver solver3;
  makeAndRunSecretnessSolver(top, solver3);
  patterns.clear();
  patterns.add<UseInitForPlaintextBranchTerminators,
               RegionBranchOpLevelInvariancePattern>(ctx, &solver3);
  walkAndApplyPatterns(top, std::move(patterns));
}

void adjustLevelsForRegionBranchOps(Operation* top) {
  LDBG() << "Adjusting levels for region branching ops";
  MLIRContext* ctx = top->getContext();
  DataFlowSolver solver;
  makeAndRunSecretnessAndLevelSolver(top, solver);

  RewritePatternSet patterns(ctx);
  patterns.add<RegionBranchOpLevelInvariancePattern>(ctx, &solver);
  walkAndApplyPatterns(top, std::move(patterns));
}

void unrollLoopsForLevelUtilization(Operation* top, int levelBudget) {
  DataFlowSolver solver;
  makeAndRunSolver(top, solver);
  MLIRContext* ctx = top->getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<PartialUnrollForLevelConsumptionAffineFor,
               PartialUnrollForLevelConsumptionSCFFor>(ctx, levelBudget,
                                                       &solver);
  walkAndApplyPatterns(top, std::move(patterns));

  LDBG() << "Deleting annotated ops";
  RewritePatternSet cleanupPatterns(ctx);
  cleanupPatterns.add<DeleteAnnotatedOps>(ctx);
  walkAndApplyPatterns(top, std::move(cleanupPatterns));
}

}  // namespace heir
}  // namespace mlir
