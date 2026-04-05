#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/MulDepthAnalysis/MulDepthAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Mgmt/Transforms/AnnotateMgmt.h"
#include "lib/Transforms/PopulateScale/PopulateScale.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/StringRef.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"             // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"           // from @llvm-project

#define DEBUG_TYPE "resolve-reconcile-ckks"

namespace mlir {
namespace heir {

namespace {

enum class CKKSReconcilePolicyKind {
  kLocalHighestMeetingPoint,
  kDefaultScaleSchedule,
};

FailureOr<CKKSReconcilePolicyKind> parseCKKSReconcilePolicy(StringRef policy) {
  if (policy == "local-highest-meeting-point") {
    return CKKSReconcilePolicyKind::kLocalHighestMeetingPoint;
  }
  if (policy == "default-scale-schedule") {
    return CKKSReconcilePolicyKind::kDefaultScaleSchedule;
  }
  return failure();
}

bool isSecretBinaryArithmetic(Operation* op) {
  return isa<arith::AddIOp, arith::AddFOp, arith::SubIOp, arith::SubFOp>(op);
}

LevelState getLevelState(Value value, DataFlowSolver& solver) {
  auto* lattice = solver.lookupState<LevelLattice>(value);
  return lattice ? lattice->getValue() : LevelState(Invalid{});
}

MulDepthState getMulDepthState(Value value, DataFlowSolver& solver) {
  auto* lattice = solver.lookupState<MulDepthLattice>(value);
  return lattice ? lattice->getValue() : MulDepthState();
}

Value materializeReconcile(IRRewriter& rewriter, Location loc, Value input,
                           int64_t inputLevel, int64_t targetLevel,
                           CKKSReconcilePolicyKind policy, int64_t id) {
  Value managed = input;

  if (inputLevel < targetLevel) {
    switch (policy) {
      case CKKSReconcilePolicyKind::kLocalHighestMeetingPoint:
        if (targetLevel - inputLevel > 1) {
          managed = mgmt::LevelReduceOp::create(rewriter, loc, managed,
                                                targetLevel - inputLevel - 1);
        }
        managed = mgmt::AdjustScaleOp::create(rewriter, loc, managed,
                                              rewriter.getI64IntegerAttr(id));
        managed = mgmt::ModReduceOp::create(rewriter, loc, managed);
        return managed;
      case CKKSReconcilePolicyKind::kDefaultScaleSchedule:
        managed = mgmt::LevelReduceOp::create(rewriter, loc, managed,
                                              targetLevel - inputLevel);
        managed = mgmt::AdjustScaleOp::create(rewriter, loc, managed,
                                              rewriter.getI64IntegerAttr(id));
        return managed;
    }
  }

  if (inputLevel == targetLevel) {
    managed = mgmt::AdjustScaleOp::create(rewriter, loc, managed,
                                          rewriter.getI64IntegerAttr(id));
  }
  return managed;
}

}  // namespace

#define GEN_PASS_DEF_RESOLVERECONCILECKKS
#include "lib/Transforms/PopulateScale/PopulateScale.h.inc"

struct ResolveReconcileCKKS
    : impl::ResolveReconcileCKKSBase<ResolveReconcileCKKS> {
  using ResolveReconcileCKKSBase::ResolveReconcileCKKSBase;

  void runOnOperation() override {
    auto parsedPolicy = parseCKKSReconcilePolicy(reconcilePolicy);
    if (failed(parsedPolicy)) {
      getOperation()->emitOpError()
          << "unsupported CKKS reconcile policy `" << reconcilePolicy
          << "`; expected `local-highest-meeting-point` or "
             "`default-scale-schedule`";
      signalPassFailure();
      return;
    }
    auto policy = *parsedPolicy;

    OpPassManager pipeline("builtin.module");
    pipeline.addPass(mgmt::createAnnotateMgmt());
    if (failed(runPipeline(pipeline, getOperation()))) {
      signalPassFailure();
      return;
    }

    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    solver.load<SecretnessAnalysis>();
    solver.load<LevelAnalysis>();
    solver.load<MulDepthAnalysis>();
    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "failed to run reconcile analyses";
      signalPassFailure();
      return;
    }

    IRRewriter rewriter(&getContext());
    SmallVector<mgmt::ReconcileOp> reconcileOps;
    getOperation()->walk(
        [&](mgmt::ReconcileOp op) { reconcileOps.push_back(op); });
    for (auto op : reconcileOps) {
      if (op->hasOneUse()) {
        continue;
      }
      SmallVector<OpOperand*> uses;
      for (OpOperand& use : op.getResult().getUses()) {
        uses.push_back(&use);
      }
      for (OpOperand* use : uses) {
        rewriter.setInsertionPoint(use->getOwner());
        auto cloned =
            mgmt::ReconcileOp::create(rewriter, op.getLoc(), op.getInput());
        use->set(cloned.getResult());
      }
      op.erase();
    }

    int64_t idCounter = 0;
    SmallVector<Operation*> consumers;
    getOperation()->walk([&](Operation* op) {
      if (isSecretBinaryArithmetic(op) && isSecret(op->getResults(), &solver)) {
        consumers.push_back(op);
      }
    });

    for (Operation* consumer : consumers) {
      struct ReconcileOperandInfo {
        unsigned operandNumber;
        mgmt::ReconcileOp reconcileOp;
        Value input;
        LevelState levelState;
        MulDepthState mulDepthState;
      };

      SmallVector<ReconcileOperandInfo, 2> reconcileOperands;
      for (OpOperand& operand : consumer->getOpOperands()) {
        auto reconcileOp = dyn_cast_if_present<mgmt::ReconcileOp>(
            operand.get().getDefiningOp());
        if (!reconcileOp) {
          continue;
        }
        reconcileOperands.push_back(ReconcileOperandInfo{
            operand.getOperandNumber(), reconcileOp, reconcileOp.getInput(),
            getLevelState(reconcileOp.getInput(), solver),
            getMulDepthState(reconcileOp.getInput(), solver)});
      }

      if (reconcileOperands.empty()) {
        continue;
      }

      LevelState targetLevelState =
          getLevelState(consumer->getResult(0), solver);
      if (targetLevelState.isMaxLevel()) {
        for (auto& operandInfo : reconcileOperands) {
          rewriter.setInsertionPoint(consumer);
          Value replacement = operandInfo.input;
          if (operandInfo.levelState.isInt()) {
            replacement = mgmt::LevelReduceMinOp::create(
                rewriter, operandInfo.reconcileOp.getLoc(), replacement);
          }
          consumer->setOperand(operandInfo.operandNumber, replacement);
          operandInfo.reconcileOp.erase();
        }
        continue;
      }

      if (targetLevelState.isInvalid() || !targetLevelState.isInt()) {
        consumer->emitOpError() << "failed to determine reconcile level";
        signalPassFailure();
        return;
      }
      int64_t targetLevel = targetLevelState.getInt();

      SmallVector<unsigned, 2> operandsToAdjust;
      for (auto& operandInfo : reconcileOperands) {
        if (operandInfo.levelState.isInt() &&
            operandInfo.levelState.getInt() < targetLevel) {
          operandsToAdjust.push_back(operandInfo.operandNumber);
        }
      }

      if (operandsToAdjust.empty() && reconcileOperands.size() == 2) {
        const auto& lhs = reconcileOperands[0];
        const auto& rhs = reconcileOperands[1];
        bool sameLevel = lhs.levelState.isInt() && rhs.levelState.isInt() &&
                         lhs.levelState.getInt() == rhs.levelState.getInt() &&
                         lhs.levelState.getInt() == targetLevel;
        bool differentMulDepth =
            lhs.mulDepthState.isInitialized() &&
            rhs.mulDepthState.isInitialized() &&
            lhs.mulDepthState.getMulDepth() != rhs.mulDepthState.getMulDepth();
        if (sameLevel && differentMulDepth) {
          operandsToAdjust.push_back(lhs.mulDepthState.getMulDepth() <
                                             rhs.mulDepthState.getMulDepth()
                                         ? lhs.operandNumber
                                         : rhs.operandNumber);
        }
      }

      for (auto& operandInfo : reconcileOperands) {
        rewriter.setInsertionPoint(consumer);
        Value replacement = operandInfo.input;
        if (llvm::is_contained(operandsToAdjust, operandInfo.operandNumber)) {
          replacement = materializeReconcile(
              rewriter, operandInfo.reconcileOp.getLoc(), operandInfo.input,
              operandInfo.levelState.getInt(), targetLevel, policy,
              idCounter++);
        }
        consumer->setOperand(operandInfo.operandNumber, replacement);
        operandInfo.reconcileOp.erase();
      }
    }

    reconcileOps.clear();
    getOperation()->walk(
        [&](mgmt::ReconcileOp op) { reconcileOps.push_back(op); });
    for (auto op : reconcileOps) {
      if (!op->use_empty()) {
        continue;
      }
      op.erase();
    }
    reconcileOps.clear();
    getOperation()->walk(
        [&](mgmt::ReconcileOp op) { reconcileOps.push_back(op); });
    for (auto op : reconcileOps) {
      Operation* consumer = *op->user_begin();
      if (!isSecretBinaryArithmetic(consumer) ||
          !isSecret(consumer->getResults(), &solver)) {
        op.getResult().replaceAllUsesWith(op.getInput());
        op.erase();
      }
    }

    OpPassManager cleanup("builtin.module");
    cleanup.addPass(createCanonicalizerPass());
    cleanup.addPass(createCSEPass());
    cleanup.addPass(mgmt::createAnnotateMgmt());
    if (failed(runPipeline(cleanup, getOperation()))) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace heir
}  // namespace mlir
