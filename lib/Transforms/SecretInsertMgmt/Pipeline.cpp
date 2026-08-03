#include "lib/Transforms/SecretInsertMgmt/Pipeline.h"

#include <algorithm>
#include <cstdint>
#include <utility>

#include "lib/Analysis/LevelAnalysis/BootstrapWaterlineAnalysis.h"
#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/MulDepthAnalysis/MulDepthAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Target/CompilationTarget/CompilationTarget.h"
#include "lib/Transforms/Halo/Patterns.h"
#include "lib/Transforms/SecretInsertMgmt/SecretInsertMgmtPatterns.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/DenseMap.h"                // from @llvm-project
#include "llvm/include/llvm/ADT/DenseSet.h"                // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"               // from @llvm-project
#include "llvm/include/llvm/ADT/SetVector.h"               // from @llvm-project
#include "llvm/include/llvm/ADT/SmallPtrSet.h"             // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "llvm/include/llvm/Support/DebugLog.h"            // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Interfaces/LoopLikeInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/include/mlir/Rewrite/FrozenRewritePatternSet.h"  // from @llvm-project
#include "mlir/include/mlir/Rewrite/PatternApplicator.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"               // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "secret-insert-mgmt"

namespace mlir {
namespace heir {

void runSolver(Operation* top, DataFlowSolver& solver) {
  if (failed(solver.initializeAndRun(top))) {
    LDBG() << "Failed to run solver!";
  }
}

void makeAndRunSolver(Operation* top, DataFlowSolver& solver,
                      int levelBudget = 40) {
  dataflow::loadBaselineAnalyses(solver);
  solver.load<SecretnessAnalysis>();
  solver.load<LevelAnalysis>(levelBudget);
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

void makeAndRunSecretnessAndLevelSolver(Operation* top, DataFlowSolver& solver,
                                        int levelBudget = 40) {
  dataflow::loadBaselineAnalyses(solver);
  solver.load<SecretnessAnalysis>();
  solver.load<LevelAnalysis>(levelBudget);
  runSolver(top, solver);
}

// Forward-simulation bootstrap placement (orion-style, concrete levels).
//
// The greedy waterline (BootstrapWaterLine) triggers on `level % waterline ==
// 0` using HEIR's LevelState lattice, which cannot reason about a deep
// activation landing a *multiply* at the bottom of the modulus chain (HEIR's
// cross-level gap, BootstrapWaterLine "TODO(#1642)"). Deep composite-sign ReLUs
// hit this: the `x*step` multiply lands at the floor where its mandatory
// post-mul rescale overflows -> a malformed ckks.rescale onto the full chain
// that fails verification. Greedy lattice-based fixes don't converge because
// the lattice reuses the isMaxLevel/Invalid sentinels for both structural
// artifacts and genuine exhaustion.
//
// Instead, simulate level consumption with our OWN concrete counter (never the
// lattice), following orion's level-DAG insight (work with concrete levels and
// a fixed budget l_eff): every value carries a "remaining levels" count; a
// fresh or just-bootstrapped value starts at l_eff; a mod_reduce (the CKKS
// rescale, the only op that drops a level) decrements it; a join (mul/add)
// takes the min of its operands. When a mod_reduce would drive an operand below
// zero (the multiply that precedes it has exhausted the chain), bootstrap that
// operand first. Processing ops in SSA order (defs precede uses) makes this a
// single forward pass; min-at-joins handles the DAG without orion's SESE
// machinery. Self-consistent on l_eff: levels stay in [0, l_eff], so param-gen
// sizes the chain to l_eff (+ the bootstrap's own internal depth).
void insertBootstrapsByForwardLevelSim(Operation* top, int lEff) {
  DataFlowSolver solver;
  makeAndRunSecretnessSolver(top, solver);

  llvm::DenseMap<Value, int> remaining;
  auto rem = [&](Value v) -> int {
    auto it = remaining.find(v);
    return it == remaining.end() ? lEff : it->second;  // default: fresh
  };

  // A "kept eval" is a multi-level-drop ReducesLevelOpInterface op (the kept
  // polynomial.eval / composite-sign stage), as opposed to mod_reduce (a
  // 1-level rescale, handled separately).
  auto isKeptEval = [](Operation* o) {
    return isa_and_nonnull<ReducesLevelOpInterface>(o) &&
           !isa<mgmt::ModReduceOp>(o);
  };
  // Total multiplicative depth of a chained-eval run starting at `start`: the
  // composite-sign `step` is 3 polynomial.evals chained (each feeds the next as
  // its reduced operand). Boot placement must treat such a run as INDIVISIBLE
  // -- bootstrapping between stages leaves the chain desynced (stage k+1
  // evaluates a different polynomial than its input was prepared for),
  // detonating the deg-27 tail. So at a chain start we size the refresh to the
  // whole run.
  auto chainTotalDrop = [&](Operation* start) -> int {
    int total = 0;
    for (Operation* cur = start; isKeptEval(cur);) {
      total += cast<ReducesLevelOpInterface>(cur).getLevelsToDrop();
      Operation* next = nullptr;
      if (cur->getNumResults() == 1) {
        Value res = cur->getResult(0);
        for (Operation* u : res.getUsers()) {
          auto rl = dyn_cast<ReducesLevelOpInterface>(u);
          if (isKeptEval(u) && rl.getOperandToReduce().get() == res) {
            next = u;
            break;
          }
        }
      }
      cur = next;
    }
    return total;
  };

  SmallVector<Operation*> ops;
  top->walk([&](Operation* op) { ops.push_back(op); });

  // Backward sweep: does this value transitively feed a mod_reduce? A value
  // at the bottom of the budget that only feeds level-preserving ops (adds,
  // rotations, the function result) needs no refresh -- skipping those
  // bootstraps is free. Reverse SSA order visits all users before the def.
  llvm::DenseSet<Value> reachesModReduce;
  for (Operation* op : llvm::reverse(ops)) {
    if (isa<mgmt::ModReduceOp>(op)) {
      for (Value operand : op->getOperands()) reachesModReduce.insert(operand);
      continue;
    }
    // A bootstrap refreshes its result, so consumption below an existing
    // bootstrap does not require its input to be refreshed.
    if (isa<mgmt::BootstrapOp>(op)) continue;
    bool resultReaches = llvm::any_of(op->getResults(), [&](Value res) {
      return reachesModReduce.contains(res);
    });
    if (resultReaches)
      for (Value operand : op->getOperands()) reachesModReduce.insert(operand);
  }

  llvm::DenseSet<Operation*> erased;

  for (Operation* op : ops) {
    if (erased.contains(op)) continue;
    if (isa<mgmt::BootstrapOp>(op)) {
      for (Value res : op->getResults()) remaining[res] = lEff;
      continue;
    }
    // mod_reduce is the level-consuming rescale; when its result lands at the
    // bottom of the budget, refresh by bootstrapping the RESULT. The rescale
    // itself stays: it is what brings the post-mul scale 2*Delta back to
    // Delta, so the bootstrap input is at the default scale -- exactly the
    // standard CKKS pattern (rescale to the floor, then bootstrap). The two
    // alternatives both fail: bootstrapping the rescale's *input* hands the
    // backend a 2*Delta ciphertext (lattigo rejects it: Q[0]/Scale below the
    // MessageRatio bound), and replacing the rescale with the bootstrap
    // leaves a Delta-deficit in the scale that every downstream ct*ct
    // squaring doubles -- the exponential negative coefficient scales seen
    // in deep Chebyshev-PS evals.
    if (auto modReduce = dyn_cast<mgmt::ModReduceOp>(op)) {
      Value in = modReduce.getInput();
      int r = rem(in) - 1;
      remaining[modReduce.getResult()] = r;
      if (r > 0) continue;
      // Refresh only pays off if something below still consumes levels.
      Value res = modReduce.getResult();
      if (!reachesModReduce.contains(res)) continue;
      OpBuilder builder(op);
      builder.setInsertionPointAfter(op);
      auto bootstrap =
          mgmt::BootstrapOp::create(builder, op->getLoc(), res.getType(), res);
      res.replaceAllUsesExcept(bootstrap.getResult(), bootstrap);
      remaining[bootstrap.getResult()] = lEff;
      // Sibling rotations of the same source each carry their own
      // rescale+refresh (matvec kernels emit one rotated copy per diagonal
      // group); rescale and rotation commute, so serve them all from this
      // ONE bootstrap by re-rotating its output. This collapses the
      // per-rotated-copy bootstraps (e.g. 2 per ReLU on ToyHELRM, ~5 per
      // ReLU on CriteoHELRM) into one refresh per logical value.
      Value src = in;
      for (Operation* user : llvm::make_early_inc_range(src.getUsers())) {
        if (user == modReduce.getOperation() || erased.contains(user)) continue;
        auto rot = dyn_cast<tensor_ext::RotateOp>(user);
        if (!rot || !rot.getOutput().hasOneUse()) continue;
        auto siblingMr =
            dyn_cast<mgmt::ModReduceOp>(*rot.getOutput().getUsers().begin());
        if (!siblingMr || erased.contains(siblingMr)) continue;
        // Already refreshed on its own (it preceded us in SSA order but
        // skipped/handled separately)? Leave it alone.
        if (llvm::any_of(siblingMr.getResult().getUsers(), [](Operation* u) {
              return isa<mgmt::BootstrapOp>(u);
            }))
          continue;
        // The replacement rotate is inserted after the bootstrap; only
        // rewrite siblings that come later, so all their users are
        // dominated by it.
        if (siblingMr->getBlock() != bootstrap->getBlock() ||
            !bootstrap->isBeforeInBlock(siblingMr))
          continue;
        builder.setInsertionPointAfter(bootstrap);
        auto newRot = tensor_ext::RotateOp::create(
            builder, rot.getLoc(), rot.getOutput().getType(),
            bootstrap.getResult(), rot.getShift());
        siblingMr.getResult().replaceAllUsesWith(newRot.getOutput());
        erased.insert(siblingMr);
        erased.insert(rot);
        siblingMr.erase();
        rot.erase();
      }
      continue;
    }
    // A kept `polynomial.eval` (lowered later to orion.chebyshev /
    // cheddar.eval_poly) consumes its whole multiplicative depth in one op
    // (ReducesLevelOpInterface::getLevelsToDrop, e.g. 4/5 for the
    // composite-sign stages). The default branch below treats every
    // non-mod_reduce op as a 0-drop pass-through, so a chained composite sign's
    // `step` would sink far below the budget with no refresh, and cross-level
    // matching can't then align it with `x` at the final `x*step` multiply
    // (TODO #1642). Count the drop, and refresh the reduced operand if the
    // chain would otherwise exhaust. (mod_reduce -- also a
    // ReducesLevelOpInterface -- was handled and `continue`d above, so only
    // multi-level-drop ops like the kept eval reach here; non-orion paths never
    // keep a polynomial.eval, so they are unaffected.)
    if (auto reduces = dyn_cast<ReducesLevelOpInterface>(op)) {
      if (op->getNumResults() == 0) continue;
      int drop = reduces.getLevelsToDrop();
      int minRem = lEff;
      bool sawSecret = false;
      for (Value operand : op->getOperands()) {
        if (isSecret(operand, &solver)) {
          minRem = std::min(minRem, rem(operand));
          sawSecret = true;
        }
      }
      int avail = sawSecret ? minRem : lEff;
      int r = avail - drop;
      Value reduced = reduces.getOperandToReduce().get();
      bool resReaches = llvm::any_of(op->getResults(), [&](Value res) {
        return reachesModReduce.contains(res);
      });
      // Refresh BEFORE the chain, not mid-chain. At a chain start (the reduced
      // operand is not itself a kept eval) require room for the WHOLE chained
      // run so all stages evaluate post-boot at healthy levels; mid-chain, fall
      // back to the per-op exhaustion check (which should not fire once the
      // chain-start refresh has).
      Operation* defOp = reduced.getDefiningOp();
      bool chainStart = !isKeptEval(defOp);
      int need = chainStart ? chainTotalDrop(op) : drop;
      if (avail - need <= 0 && isSecret(reduced, &solver) && resReaches) {
        OpBuilder builder(op);
        builder.setInsertionPoint(op);
        auto bootstrap = mgmt::BootstrapOp::create(builder, op->getLoc(),
                                                   reduced.getType(), reduced);
        op->replaceUsesOfWith(reduced, bootstrap.getResult());
        remaining[bootstrap.getResult()] = lEff;
        r = lEff - drop;
      }
      for (Value res : op->getResults()) {
        if (isSecret(res, &solver)) remaining[res] = r;
      }
      continue;
    }
    // Everything else: result remaining = min over secret operands
    // (level_reduce and relinearize don't drop levels in this model;
    // min-at-joins is exact for the placement decision since cross-level
    // matching aligns to the min).
    if (op->getNumResults() == 0) continue;
    int r = lEff;
    bool sawSecret = false;
    for (Value operand : op->getOperands()) {
      if (isSecret(operand, &solver)) {
        r = std::min(r, rem(operand));
        sawSecret = true;
      }
    }
    for (Value res : op->getResults()) {
      if (isSecret(res, &solver)) remaining[res] = sawSecret ? r : lEff;
    }
  }
}

LogicalResult runInsertMgmtPipeline(Operation* top,
                                    const InsertMgmtPipelineOptions& options) {
  LDBG(2) << "Starting insert-mgmt pipeline";
  peelPlaintextIterations(top);
  LLVM_DEBUG(top->dump());

  insertMgmtInitForPlaintexts(top, options.includeFloats);
  LLVM_DEBUG(top->dump());

  LDBG(2) << "Inserting mod reduce";
  insertModReduceBeforeOrAfterMult(top, options.modReduceAfterMul,
                                   options.modReduceBeforeMulIncludeFirstMul,
                                   options.includeFloats);
  LLVM_DEBUG(top->dump());

  // this must be run after ModReduceAfterMult
  LDBG(2) << "Inserting relinearize";
  insertRelinearizeAfterMult(top, options.includeFloats);

  ModuleOp module = dyn_cast<ModuleOp>(top);
  if (!module) {
    module = top->getParentOfType<ModuleOp>();
  }
  FailureOr<CompilationTarget> target = failure();
  if (module) {
    target = getTargetConfig(module);
  }

  int64_t bootstrapLevelsConsumed = 0;
  int64_t maxLevel = 40;
  if (succeeded(target)) {
    bootstrapLevelsConsumed = target->bootstrapLevelsConsumed;
  }

  int budget = options.levelBudget == -1
                   ? maxLevel
                   : (options.levelBudget + bootstrapLevelsConsumed);

  // Run Level Analysis to check for convergence
  DataFlowSolver levelSolver;
  makeAndRunSolver(top, levelSolver, budget);

  auto nonInvariantLoops = getNonInvariantLoops(top, &levelSolver);

  LDBG(2) << "Found " << nonInvariantLoops.size() << " non-invariant loops";
  for (auto* loop : nonInvariantLoops) {
    LDBG(2) << "Processing non-invariant loop " << *loop;
    DataFlowSolver secretnessSolver;
    makeAndRunSecretnessSolver(top, secretnessSolver);
    if (options.bootstrapWaterline.has_value()) {
      bootstrapLoopIterArgs(loop, &secretnessSolver);
    }

    DataFlowSolver freshLevelSolver;
    makeAndRunSolver(top, freshLevelSolver, budget);
    unrollLoopForLevelUtilization(loop, &freshLevelSolver, budget);
  }

  makeRegionBranchOpsLevelInvariant(top);

  int idCounter = 0;  // for making adjust_scale op different to avoid cse
  if (options.bootstrapWaterline.has_value()) {
    LDBG(2) << "Bootstrap placement (forward level simulation)";
    // The effective per-refresh depth (l_eff) must respect BOTH knobs: the
    // waterline (how many levels a value may consume between refreshes) and
    // the level budget (how many compute levels the chain provides, and the
    // bound above which LevelAnalysis marks the lattice Invalid -- an Invalid
    // lattice makes the cross-level patterns silently skip ops, ending in
    // ring-mismatch verification failures in SecretToCKKS). Deep configs
    // (e.g. composite-sign ReLU with waterline 25) must therefore raise
    // greedy-level-budget alongside greedy-bootstrap-waterline.
    int64_t lEff = options.bootstrapWaterline.value();
    if (options.levelBudget > 0)
      lEff = std::min<int64_t>(lEff, options.levelBudget);
    insertBootstrapsByForwardLevelSim(top, lEff);
  }

  // An if statement must have each branch producing the same level as a result,
  // so the branch with the higher level must insert a level_reduce op.
  adjustLevelsForRegionBranchOps(top, budget);
  adjustScalesForRegionBranchOps(top, &idCounter);

  LDBG(2) << "Handling cross level ops";
  handleCrossLevelOps(top, &idCounter, options.includeFloats, budget,
                      options.cheddarMode);

  LDBG(2) << "Handling cross mul depth ops";
  // A same-level mul-depth mismatch only represents a scale mismatch for the
  // rescale-before-multiply policy that excludes the first multiply. With
  // rescale-after-multiply, every multiply result has already returned to the
  // canonical scale for its new level; MulDepthAnalysis still retaining depth
  // one across mgmt.modreduce is bookkeeping, not a real scale difference.
  // Inserting adjust_scale for that phantom mismatch can make an otherwise
  // solvable cross-level adjustment underdetermined.
  if (!options.modReduceAfterMul &&
      !options.modReduceBeforeMulIncludeFirstMul) {
    handleCrossMulDepthOps(top, &idCounter, options.includeFloats, budget,
                           options.cheddarMode);
  }

  // An if statement must have each branch producing the same level as a result,
  // so the branch with the higher level must insert a level_reduce op.
  adjustLevelsForRegionBranchOps(top, budget);

  // Clean up invalid modreduces and other redundant mgmt ops before final
  // analysis.
  {
    MLIRContext* ctx = top->getContext();
    RewritePatternSet cleanupPatterns(ctx);
    mgmt::ModReduceOp::getCanonicalizationPatterns(cleanupPatterns, ctx);
    mgmt::LevelReduceOp::getCanonicalizationPatterns(cleanupPatterns, ctx);
    mgmt::AdjustScaleOp::getCanonicalizationPatterns(cleanupPatterns, ctx);
    mgmt::LevelReduceMinOp::getCanonicalizationPatterns(cleanupPatterns, ctx);
    (void)walkAndApplyPatterns(top, std::move(cleanupPatterns));
  }

  DataFlowSolver finalSolver;
  makeAndRunSolver(top, finalSolver, budget);
  return validateLevelAnalysis(finalSolver, top);
}

void insertMgmtInitForPlaintexts(Operation* top, bool includeFloats) {
  LDBG(2) << "Inserting mgmt.init";
  DataFlowSolver solver;
  makeAndRunSecretnessSolver(top, solver);

  MLIRContext* ctx = top->getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<UseInitOpForPlaintextOperand<arith::AddIOp>,
               UseInitOpForPlaintextOperand<arith::SubIOp>,
               UseInitOpForPlaintextOperand<arith::MulIOp>,
               // Kept (lintrans) rotate_and_reduce: its plaintext diagonals
               // operand needs an mgmt.init like any other plaintext operand.
               UseInitOpForPlaintextOperand<tensor_ext::RotateAndReduceOp>,
               UseInitOpForPlaintextOperand<tensor::ExtractSliceOp>,
               UseInitOpForPlaintextOperand<tensor::InsertSliceOp>,
               UseInitOpForPlaintextOperand<tensor::InsertOp>>(ctx, top,
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
    // Kept (lintrans) rotate_and_reduce is a ct x pt matrix multiply:
    // exactly one rescale after it.
    patterns.add<ModReduceAfterMult<tensor_ext::RotateAndReduceOp>>(ctx, top,
                                                                    &solver);
    if (includeFloats)
      patterns.add<ModReduceAfterMult<arith::MulFOp>>(ctx, top, &solver);
  } else {
    patterns.add<ModReduceBefore<arith::MulIOp>>(ctx, beforeMulIncludeFirstMul,
                                                 top, &solver);
    // Kept (lintrans) rotate_and_reduce behaves like a ct x pt mult under
    // rescale-before-mult policies too.
    patterns.add<ModReduceBefore<tensor_ext::RotateAndReduceOp>>(
        ctx, beforeMulIncludeFirstMul, top, &solver);
    if (includeFloats)
      patterns.add<ModReduceBefore<arith::MulFOp>>(
          ctx, beforeMulIncludeFirstMul, top, &solver);
    // includeFirstMul = false here
    // as before yield we only want mulResult to be mod reduced
    patterns.add<ModReduceBefore<secret::YieldOp>>(
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
                         int levelBudget, bool cheddarMode) {
  DataFlowSolver solver;
  makeAndRunSecretnessAndLevelSolver(top, solver, levelBudget);
  MLIRContext* ctx = top->getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<MatchCrossLevel<arith::AddIOp>, MatchCrossLevel<arith::SubIOp>,
               MatchCrossLevel<arith::MulIOp>,
               MatchCrossLevel<tensor::InsertSliceOp>,
               MatchCrossLevel<tensor::InsertOp>>(ctx, idCounter, top, &solver,
                                                  cheddarMode);
  if (includeFloats)
    patterns.add<MatchCrossLevel<arith::AddFOp>, MatchCrossLevel<arith::SubFOp>,
                 MatchCrossLevel<arith::MulFOp>>(ctx, idCounter, top, &solver,
                                                 cheddarMode);
  (void)walkAndApplyPatterns(top, std::move(patterns));
}

// This only happens for the before-mul but not include-first-mul case. At the
// first level, a Value can be a mul result while its peer is not, so match
// their scales by adding one adjust_scale op. runInsertMgmtPipeline guards
// this helper by that policy; keep the Cheddar check for direct callers.
void handleCrossMulDepthOps(Operation* top, int* idCounter, bool includeFloats,
                            int levelBudget, bool cheddarMode) {
  // Cheddar uses rescale-after-mult and a fixed canonical scale per level, so
  // every ciphertext is always at its level's canonical scale. By the time we
  // get here, handleCrossLevelOps has already aligned operand levels with
  // level_reduce, so the two operands of any add/sub share a level and hence
  // share the canonical scale -- there is no same-level scale mismatch to fix.
  // The MulDepthAnalysis can still report a {0,1} "cross mul depth" because
  // mgmt.modreduce does not reset the mul-depth lattice (only mgmt.bootstrap
  // does), so a rescaled first-mul-after-bootstrap result still reads as depth
  // 1. Emitting adjust_scale for that phantom mismatch would actually corrupt
  // the scale (and Cheddar rejects adjust_scale outright), so skip it.
  if (cheddarMode) return;

  DataFlowSolver solver;
  makeAndRunSolver(top, solver, levelBudget);
  MLIRContext* ctx = top->getContext();
  RewritePatternSet patterns(ctx);
  patterns
      .add<MatchCrossMulDepth<arith::AddIOp>, MatchCrossMulDepth<arith::SubIOp>,
           MatchCrossMulDepth<arith::MulIOp>,
           MatchCrossMulDepth<tensor::InsertSliceOp>,
           MatchCrossMulDepth<tensor::InsertOp>>(ctx, idCounter, top, &solver);
  if (includeFloats)
    patterns.add<MatchCrossMulDepth<arith::AddFOp>,
                 MatchCrossMulDepth<arith::SubFOp>,
                 MatchCrossMulDepth<arith::MulFOp>>(ctx, idCounter, top,
                                                    &solver);
  (void)walkAndApplyPatterns(top, std::move(patterns));
}

void insertBootstrapWaterLine(Operation* top, int bootstrapWaterline,
                              int levelBudget, int bootstrapLevelsConsumed,
                              bool includeFloats, int* idCounter) {
  DataFlowSolver solver;
  dataflow::loadBaselineAnalyses(solver);
  solver.load<SecretnessAnalysis>();
  solver.load<BootstrapWaterlineAnalysis>(bootstrapWaterline, levelBudget,
                                          bootstrapLevelsConsumed);
  if (failed(solver.initializeAndRun(top))) {
    LDBG() << "Failed to run solver!";
    return;
  }

  // Walk the IR to collect all Values that need bootstrapping
  SmallVector<Value> targets;
  llvm::SmallPtrSet<Value, 16> visited;
  walkValues(top, [&](Value value) {
    if (!visited.insert(value).second) {
      return;
    }
    if (isa<secret::SecretType>(value.getType())) {
      return;
    }
    if (auto* lattice = solver.lookupState<BootstrapWaterlineLattice>(value)) {
      if (lattice->getValue().getNeedsBootstrap()) {
        targets.push_back(value);
      }
    }
  });

  // Mutate the IR
  OpBuilder builder(top->getContext());
  for (Value needsBootstrap : targets) {
    Operation* defOp = needsBootstrap.getDefiningOp();
    if (defOp && isa<ReducesLevelOpInterface>(defOp)) {
      // To defer bootstraps as long as possible, we mark the op result of a
      // modreduce op as needing bootstrap when that modreduce would reduce the
      // level to less than zero. So instead we need to bootstrap the operand of
      // the modreduce op.
      auto reduceOp = cast<ReducesLevelOpInterface>(defOp);
      Value operandToReduce = reduceOp.getOperandToReduce().get();
      builder.setInsertionPoint(defOp);
      auto bootstrapOp = mgmt::BootstrapOp::create(
          builder, defOp->getLoc(), operandToReduce.getType(), operandToReduce);

      Value newOperand = bootstrapOp.getResult();
      if (includeFloats && isa<mgmt::ModReduceOp>(defOp)) {
        // In CKKS, bootstrap resets the scale to Delta. If the next operation
        // is a modreduce, it will drop the scale by Delta, resulting in a scale
        // of 0. To prevent this, we insert an adjust_scale op here to raise the
        // scale to Delta^2 before the modreduce, so the modreduce output scale
        // remains at Delta.
        auto adjustScaleOp = mgmt::AdjustScaleOp::create(
            builder, defOp->getLoc(), newOperand,
            builder.getI64IntegerAttr((*idCounter)++));
        newOperand = adjustScaleOp.getResult();
      }
      reduceOp.getOperandToReduce().set(newOperand);
    } else {
      if (defOp) {
        builder.setInsertionPointAfter(defOp);
      } else {
        BlockArgument blockArg = cast<BlockArgument>(needsBootstrap);
        builder.setInsertionPointToStart(blockArg.getOwner());
      }
      auto bootstrapOp =
          mgmt::BootstrapOp::create(builder, needsBootstrap.getLoc(),
                                    needsBootstrap.getType(), needsBootstrap);
      needsBootstrap.replaceAllUsesExcept(bootstrapOp.getResult(), bootstrapOp);
    }
  }
  // makeAndRunSolver(top, solver, levelBudget);
  // MLIRContext* ctx = top->getContext();
  // RewritePatternSet patterns(ctx);
  // patterns.add<BootstrapWaterLine<mgmt::ModReduceOp>>(ctx, top, &solver,
  //                                                     bootstrapWaterline);
  // (void)walkAndApplyPatterns(top, std::move(patterns));
}

void peelPlaintextIterations(Operation* top) {
  LDBG(2) << "Peeling plaintext iterations";
  MLIRContext* ctx = top->getContext();
  DataFlowSolver solver;
  makeAndRunSecretnessSolver(top, solver);
  RewritePatternSet patterns(ctx);
  patterns.add<PeelPlaintextAffineForInit, PeelPlaintextScfForInit>(ctx,
                                                                    &solver);
  walkAndApplyPatterns(top, std::move(patterns));
}

void bootstrapLoopIterArgs(Operation* loopOp, DataFlowSolver* solver) {
  LDBG(2) << "Bootstrapping loop iter args";
  MLIRContext* ctx = loopOp->getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<BootstrapIterArgsPattern<affine::AffineForOp>,
               BootstrapIterArgsPattern<scf::ForOp>>(ctx, solver);
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  PatternApplicator applicator(frozenPatterns);
  applicator.applyDefaultCostModel();

  PatternRewriter rewriter(ctx);
  (void)applicator.matchAndRewrite(loopOp, rewriter);
}

void makeRegionBranchOpsLevelInvariant(Operation* top) {
  LDBG(2) << "Making region branch ops level invariant";
  MLIRContext* ctx = top->getContext();
  DataFlowSolver solver;
  makeAndRunSecretnessSolver(top, solver);
  RewritePatternSet patterns(ctx);
  patterns.add<UseInitForPlaintextBranchTerminators,
               RegionBranchOpLevelInvariancePattern>(ctx, &solver);
  walkAndApplyPatterns(top, std::move(patterns));
}

SmallVector<Operation*> getNonInvariantLoops(Operation* top,
                                             DataFlowSolver* solver) {
  LDBG(2) << "Getting non-invariant loops";
  SmallVector<Operation*> nonInvariantLoops;

  auto isInvariant = [&](LoopLikeOpInterface forOp) {
    for (auto [i, iterArg] : llvm::enumerate(forOp.getRegionIterArgs())) {
      if (!isSecret(iterArg, solver)) continue;

      auto* initLattice =
          solver->lookupState<LevelLattice>(forOp.getInits()[i]);
      auto yieldedValue =
          forOp.getTiedLoopYieldedValue(cast<BlockArgument>(iterArg))->get();
      auto* yieldLattice = solver->lookupState<LevelLattice>(yieldedValue);

      if (!initLattice || !yieldLattice || !initLattice->getValue().isInt() ||
          !yieldLattice->getValue().isInt()) {
        return false;
      }

      if (initLattice->getValue().getInt() !=
          yieldLattice->getValue().getInt()) {
        return false;
      }
    }
    return true;
  };

  // Post-order walk means we process nested loops from the inside out.
  top->walk<WalkOrder::PostOrder>([&](Operation* op) {
    if (isa<affine::AffineForOp, scf::ForOp>(op)) {
      if (!isInvariant(cast<LoopLikeOpInterface>(op))) {
        nonInvariantLoops.push_back(op);
      }
    }
  });

  return nonInvariantLoops;
}

void adjustLevelsForRegionBranchOps(Operation* top, int levelBudget) {
  LDBG(2) << "Adjusting levels for region branching ops";
  MLIRContext* ctx = top->getContext();
  DataFlowSolver solver;
  makeAndRunSecretnessAndLevelSolver(top, solver, levelBudget);

  RewritePatternSet patterns(ctx);
  patterns.add<RegionBranchOpLevelInvariancePattern>(ctx, &solver);
  walkAndApplyPatterns(top, std::move(patterns));
}

void adjustScalesForRegionBranchOps(Operation* top, int* idCounter) {
  LDBG(2) << "Adjusting scales for region branching ops";
  MLIRContext* ctx = top->getContext();
  DataFlowSolver solver;
  makeAndRunSecretnessAndMulDepthSolver(top, solver);

  RewritePatternSet patterns(ctx);
  patterns.add<RegionBranchOpScaleInvariancePattern>(ctx, &solver, idCounter);
  walkAndApplyPatterns(top, std::move(patterns));
}

void unrollLoopForLevelUtilization(Operation* loopOp, DataFlowSolver* solver,
                                   int levelBudget) {
  MLIRContext* ctx = loopOp->getContext();
  PatternRewriter rewriter(ctx);
  Operation* parentOp = loopOp->getParentOp();

  // A pattern driver is not appropriate here because we need to unroll
  // the loops from inner-most to outer-most. The order in which nested
  // loops are returned from getNonInvariantLoops ensures this.
  TypeSwitch<Operation*>(loopOp)
      .Case<affine::AffineForOp, scf::ForOp>([&](auto op) {
        (void)doPartialUnroll(op, rewriter, levelBudget, solver);
      })
      .Default([&](auto op) {
        LDBG(2) << "Unknown loop type " << loopOp->getName();
      });

  LDBG(2) << "Deleting annotated ops";
  RewritePatternSet cleanupPatterns(ctx);
  cleanupPatterns.add<DeleteAnnotatedOps>(ctx);
  walkAndApplyPatterns(parentOp, std::move(cleanupPatterns));
}

}  // namespace heir
}  // namespace mlir
