#include <cmath>
#include <utility>

#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/ScaleAnalysis/ScaleAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Mgmt/Transforms/AnnotateMgmt.h"
#include "lib/Transforms/PopulateScale/PopulateScalePatterns.h"
#include "lib/Utils/APIntUtils.h"
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "llvm/include/llvm/Support/DebugLog.h"            // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"             // from @llvm-project
#include "mlir/include/mlir/IR/SymbolTable.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"           // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

// IWYU pragma: begin_keep
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Parameters/CKKS/Params.h"
#include "lib/Transforms/PopulateScale/PopulateScale.h"
// IWYU pragma: end_keep

#define DEBUG_TYPE "populate-scale-ckks"

namespace mlir {
namespace heir {

enum class CKKSScalePolicyKind { kNominal, kPrecise };

FailureOr<CKKSScalePolicyKind> parseCKKSScalePolicy(StringRef scalePolicy) {
  if (scalePolicy == kCKKSNominalScalePolicyValue) {
    return CKKSScalePolicyKind::kNominal;
  }
  if (scalePolicy == kCKKSPreciseScalePolicyValue) {
    return CKKSScalePolicyKind::kPrecise;
  }
  return failure();
}

class CKKSAdjustScaleMaterializer : public AdjustScaleMaterializer {
 public:
  CKKSAdjustScaleMaterializer(ckks::SchemeParamAttr schemeParamAttr,
                              CKKSScalePolicyKind scalePolicy)
      : schemeParamAttr(schemeParamAttr), scalePolicy(scalePolicy) {}

  virtual ~CKKSAdjustScaleMaterializer() = default;

  FailureOr<APInt> deltaScale(mgmt::AdjustScaleOp op, const APInt& scale,
                              const APInt& inputScale) const override {
    auto directDelta = divideUnsignedAPIntExact(scale, inputScale);
    if (succeeded(directDelta)) {
      return *directDelta;
    }
    if (scalePolicy != CKKSScalePolicyKind::kPrecise || !op->hasOneUse()) {
      return failure();
    }

    Operation* consumer = *op->user_begin();
    SmallVector<APInt> droppedPrimes;
    Value finalAdjustedValue = op.getResult();
    while (auto modReduceOp = dyn_cast<mgmt::ModReduceOp>(consumer)) {
      auto inputMgmtAttr =
          mgmt::findMgmtAttrAssociatedWith(modReduceOp.getInput());
      if (!inputMgmtAttr) {
        return failure();
      }
      int level = inputMgmtAttr.getLevel();
      auto droppedPrimeAttrs = schemeParamAttr.getQ().asArrayRef();
      if (level < 0 || level >= static_cast<int>(droppedPrimeAttrs.size())) {
        return failure();
      }
      droppedPrimes.push_back(
          APInt(64, static_cast<uint64_t>(droppedPrimeAttrs[level])));
      finalAdjustedValue = modReduceOp.getResult();
      if (!modReduceOp->hasOneUse()) {
        break;
      }
      consumer = *modReduceOp->user_begin();
    }
    if (droppedPrimes.empty()) {
      return failure();
    }

    auto finalAdjustedMgmtAttr =
        mgmt::findMgmtAttrAssociatedWith(finalAdjustedValue);
    if (!finalAdjustedMgmtAttr || !finalAdjustedMgmtAttr.getScale()) {
      return failure();
    }
    return solveUnsignedPostRescaleScaleDeltaChain(
        inputScale, mgmt::getScaleAsAPInt(finalAdjustedMgmtAttr),
        droppedPrimes);
  }

 private:
  ckks::SchemeParamAttr schemeParamAttr;
  CKKSScalePolicyKind scalePolicy;
};

#define GEN_PASS_DEF_POPULATESCALECKKS
#include "lib/Transforms/PopulateScale/PopulateScale.h.inc"

template <typename ScaleModelT>
LogicalResult createAndRunDataflowForPolicy(
    Operation* op, DataFlowSolver& solver, int64_t logDefaultScale,
    ckks::SchemeParamAttr ckksSchemeParamAttr, bool beforeMulIncludeFirstMul) {
  SymbolTableCollection symbolTable;
  dataflow::loadBaselineAnalyses(solver);
  solver.load<SecretnessAnalysis>();
  APInt inputScale = getNominalPowerOfTwoScaleFromLog2(logDefaultScale);
  if (beforeMulIncludeFirstMul && !moduleIsCheddar(op)) {
    LDBG() << "Encoding at scale^2 due to 'include-first-mul' config";
    inputScale = multiplyUnsignedAPIntExact(inputScale, inputScale);
  } else if (beforeMulIncludeFirstMul && moduleIsCheddar(op)) {
    LDBG() << "Keeping CHEDDAR inputs on the canonical scale bucket";
  }
  auto param = ckks::getSchemeParamFromAttr(ckksSchemeParamAttr);
  solver.load<ScaleAnalysis<ScaleModelT>>(param,
                                          /*inputScale*/ inputScale);
  // Back-prop ScaleAnalysis depends on (forward) ScaleAnalysis
  solver.load<ScaleAnalysisBackward<ScaleModelT>>(symbolTable, param);

  return solver.initializeAndRun(op);
}

LogicalResult createAndRunDataflow(Operation* op, DataFlowSolver& solver,
                                   int64_t logDefaultScale,
                                   ckks::SchemeParamAttr ckksSchemeParamAttr,
                                   bool beforeMulIncludeFirstMul,
                                   CKKSScalePolicyKind scalePolicy) {
  switch (scalePolicy) {
    case CKKSScalePolicyKind::kNominal:
      return createAndRunDataflowForPolicy<CKKSScaleModel>(
          op, solver, logDefaultScale, ckksSchemeParamAttr,
          beforeMulIncludeFirstMul);
    case CKKSScalePolicyKind::kPrecise:
      return createAndRunDataflowForPolicy<CKKSPreciseScaleModel>(
          op, solver, logDefaultScale, ckksSchemeParamAttr,
          beforeMulIncludeFirstMul);
  }
  return failure();
}

FailureOr<ckks::SchemeParamAttr> maybeRegeneratePreciseSchemeParam(
    Operation* moduleOp, ckks::SchemeParamAttr schemeParamAttr) {
  auto maxLevel = getMaxLevel(moduleOp);
  if (!maxLevel.has_value()) {
    return schemeParamAttr;
  }

  auto currentSchemeParam = ckks::getSchemeParamFromAttr(schemeParamAttr);
  if (currentSchemeParam.getLevel() >= maxLevel.value()) {
    return schemeParamAttr;
  }

  // This pass still runs on secret/mgmt IR, before SecretToCKKS embeds the
  // selected modulus chain into final CKKS/LWE ciphertext types. Regenerating
  // the scheme parameter here therefore updates the authoritative module-level
  // CKKS params before any backend-facing types are materialized.

  auto q = schemeParamAttr.getQ().asArrayRef();
  if (q.empty()) {
    return failure();
  }

  int firstModBits =
      APInt(64, static_cast<uint64_t>(q.front())).getActiveBits();
  int scalingModBits = schemeParamAttr.getLogDefaultScale();
  int slotNumber = 0;
  if (auto requestedSlotCount =
          moduleOp->getAttrOfType<IntegerAttr>(kRequestedSlotCountAttrName)) {
    slotNumber = requestedSlotCount.getInt();
  } else if (auto actualSlotCount = moduleOp->getAttrOfType<IntegerAttr>(
                 kActualSlotCountAttrName)) {
    slotNumber = actualSlotCount.getInt();
  }
  bool usePublicKey =
      schemeParamAttr.getEncryptionType() == ckks::CKKSEncryptionType::pk;
  bool encryptionTechniqueExtended = schemeParamAttr.getEncryptionTechnique() ==
                                     ckks::CKKSEncryptionTechnique::extended;
  bool reducedError = false;
  if (auto reducedErrorAttr =
          moduleOp->getAttrOfType<BoolAttr>(kCKKSReducedErrorAttrName)) {
    reducedError = reducedErrorAttr.getValue();
  }

  auto newSchemeParam = ckks::SchemeParam::getConcreteSchemeParam(
      firstModBits, scalingModBits, maxLevel.value(), slotNumber, usePublicKey,
      encryptionTechniqueExtended, reducedError);
  auto* context = moduleOp->getContext();
  OpBuilder builder(context);
  moduleOp->setAttr(
      ckks::CKKSDialect::kSchemeParamAttrName,
      ckks::SchemeParamAttr::get(
          context, log2(newSchemeParam.getRingDim()),
          DenseI64ArrayAttr::get(context, ArrayRef(newSchemeParam.getQi())),
          DenseI64ArrayAttr::get(context, ArrayRef(newSchemeParam.getPi())),
          newSchemeParam.getLogDefaultScale(),
          usePublicKey ? ckks::CKKSEncryptionType::pk
                       : ckks::CKKSEncryptionType::sk,
          encryptionTechniqueExtended
              ? ckks::CKKSEncryptionTechnique::extended
              : ckks::CKKSEncryptionTechnique::standard));
  moduleOp->setAttr(kRequestedSlotCountAttrName,
                    builder.getI64IntegerAttr(slotNumber));
  moduleOp->setAttr(kActualSlotCountAttrName,
                    builder.getI64IntegerAttr(newSchemeParam.getRingDim() / 2));
  return moduleOp->getAttrOfType<ckks::SchemeParamAttr>(
      ckks::CKKSDialect::kSchemeParamAttrName);
}

bool isScaleBalancingConsumer(Operation* op) {
  return isa<arith::AddIOp, arith::AddFOp, arith::SubIOp, arith::SubFOp,
             arith::MulIOp, arith::MulFOp>(op);
}

LogicalResult lowerAdjustScaleMeetingPoint(mgmt::AdjustScaleOp op) {
  if (!op->hasOneUse()) {
    return op.emitOpError()
           << "precise scale refinement requires adjust_scale to have one use";
  }

  Value valueToLower = op.getResult();
  Operation* consumer = *op->user_begin();
  while (auto modReduceOp = dyn_cast<mgmt::ModReduceOp>(consumer)) {
    if (!modReduceOp->hasOneUse()) {
      return modReduceOp.emitOpError()
             << "precise scale refinement requires modreduce chain to have one "
                "use";
    }
    valueToLower = modReduceOp.getResult();
    consumer = *modReduceOp->user_begin();
  }

  if (!isScaleBalancingConsumer(consumer)) {
    return op.emitOpError()
           << "precise scale refinement only supports adjust_scale feeding a "
              "binary secret arithmetic op";
  }

  Value peerOperand;
  for (Value operand : consumer->getOperands()) {
    if (operand != valueToLower) {
      peerOperand = operand;
      break;
    }
  }
  if (!peerOperand) {
    return op.emitOpError()
           << "failed to identify peer operand for precise scale refinement";
  }

  IRRewriter rewriter(op.getContext());
  rewriter.setInsertionPoint(consumer);
  auto loweredAdjusted =
      mgmt::ModReduceOp::create(rewriter, consumer->getLoc(), valueToLower);
  auto loweredPeer =
      mgmt::ModReduceOp::create(rewriter, consumer->getLoc(), peerOperand);
  consumer->replaceUsesOfWith(valueToLower, loweredAdjusted.getResult());
  consumer->replaceUsesOfWith(peerOperand, loweredPeer.getResult());
  return success();
}

FailureOr<bool> refinePreciseAdjustScales(
    Operation* top, const CKKSAdjustScaleMaterializer& mat) {
  SmallVector<mgmt::AdjustScaleOp> adjustScaleOps;
  top->walk([&](mgmt::AdjustScaleOp adjustScaleOp) {
    adjustScaleOps.push_back(adjustScaleOp);
  });

  for (auto adjustScaleOp : adjustScaleOps) {
    auto adjustScaleMgmtAttr =
        mgmt::findMgmtAttrAssociatedWith(adjustScaleOp.getResult());
    auto inputMgmtAttr =
        mgmt::findMgmtAttrAssociatedWith(adjustScaleOp.getInput());
    if (!adjustScaleMgmtAttr || !inputMgmtAttr ||
        !adjustScaleMgmtAttr.getScale() || !inputMgmtAttr.getScale()) {
      continue;
    }

    APInt targetScale = mgmt::getScaleAsAPInt(adjustScaleMgmtAttr);
    APInt inputScale = mgmt::getScaleAsAPInt(inputMgmtAttr);
    if (APInt::isSameValue(targetScale, inputScale, /*SignedCompare=*/false)) {
      continue;
    }
    if (succeeded(mat.deltaScale(adjustScaleOp, targetScale, inputScale))) {
      continue;
    }
    if (failed(lowerAdjustScaleMeetingPoint(adjustScaleOp))) {
      return failure();
    }
    return true;
  }
  return false;
}

struct PopulateScaleCKKS : impl::PopulateScaleCKKSBase<PopulateScaleCKKS> {
  using PopulateScaleCKKSBase::PopulateScaleCKKSBase;

  void runOnOperation() override {
    auto parsedScalePolicy = parseCKKSScalePolicy(scalePolicy);
    if (failed(parsedScalePolicy)) {
      getOperation()->emitOpError()
          << "unsupported CKKS scale policy `" << scalePolicy
          << "`; expected `nominal` or `precise`";
      signalPassFailure();
      return;
    }
    moduleSetCKKSScalePolicy(getOperation(), scalePolicy);
    bool useQAwarePreciseScalePolicy =
        moduleUsesQAwarePreciseCKKSScalePolicy(getOperation());
    CKKSScalePolicyKind effectiveScalePolicyKind =
        useQAwarePreciseScalePolicy ? CKKSScalePolicyKind::kPrecise
                                    : CKKSScalePolicyKind::kNominal;

    auto ckksSchemeParamAttr = mlir::dyn_cast<ckks::SchemeParamAttr>(
        getOperation()->getAttr(ckks::CKKSDialect::kSchemeParamAttrName));
    auto logDefaultScale = ckksSchemeParamAttr.getLogDefaultScale();

    auto runAnnotateMgmt = [&]() -> LogicalResult {
      OpPassManager normalizeMgmt("builtin.module");
      normalizeMgmt.addPass(mgmt::createAnnotateMgmt());
      return runPipeline(normalizeMgmt, getOperation());
    };

    auto validateAdjustScaleResolution =
        [&](DataFlowSolver& solver,
            SmallVectorImpl<mgmt::AdjustScaleOp>& noOpAdjustScales)
        -> LogicalResult {
      bool scaleConflict = false;
      getOperation()->walk([&](mgmt::AdjustScaleOp op) {
        auto* lattice = solver.lookupState<ScaleLattice>(op.getResult());
        if (!lattice) {
          op.emitOpError() << "dataflow analysis failed to populate scale "
                              "lattice for result";
          scaleConflict = true;
          return WalkResult::interrupt();
        }
        if (lattice->getValue().hasConflict()) {
          op.emitOpError() << "dataflow analysis found conflicting exact "
                              "scales for result";
          scaleConflict = true;
          return WalkResult::interrupt();
        }
        if (!lattice->getValue().isInitialized()) {
          auto* inputLattice = solver.lookupState<ScaleLattice>(op.getInput());
          if (inputLattice && inputLattice->getValue().isInitialized() &&
              !inputLattice->getValue().hasConflict()) {
            noOpAdjustScales.push_back(op);
            return WalkResult::advance();
          }
          op.emitOpError() << "dataflow analysis did not resolve a target "
                              "scale";
          scaleConflict = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      return failure(scaleConflict);
    };

    SmallVector<mgmt::AdjustScaleOp> noOpAdjustScales;
    bool refinementConverged = false;
    int maxScaleRefinementIterations = 1;
    if (useQAwarePreciseScalePolicy) {
      int adjustScaleCount = 0;
      getOperation()->walk([&](mgmt::AdjustScaleOp) { ++adjustScaleCount; });
      auto maxLevel = getMaxLevel(getOperation()).value_or(0);
      // Each successful refinement lowers at least one adjust_scale meeting
      // point by one level, so the total number of meaningful rewrites is
      // bounded by the number of adjust_scale sites times the available levels
      // below them. Add a small cushion for the final fixed-point iteration.
      maxScaleRefinementIterations =
          std::max(1, adjustScaleCount * (maxLevel + 1) + 1);
    }

    for (int refinementIteration = 0;
         refinementIteration < maxScaleRefinementIterations;
         ++refinementIteration) {
      if (failed(runAnnotateMgmt())) {
        signalPassFailure();
        return;
      }

      if (useQAwarePreciseScalePolicy) {
        auto regeneratedSchemeParam = maybeRegeneratePreciseSchemeParam(
            getOperation(), ckksSchemeParamAttr);
        if (failed(regeneratedSchemeParam)) {
          getOperation()->emitOpError()
              << "failed to regenerate CKKS parameters for precise scale "
                 "refinement";
          signalPassFailure();
          return;
        }
        ckksSchemeParamAttr = *regeneratedSchemeParam;
      }
      CKKSAdjustScaleMaterializer materializer(ckksSchemeParamAttr,
                                               effectiveScalePolicyKind);

      DataFlowSolver solver;
      if (failed(createAndRunDataflow(
              getOperation(), solver, logDefaultScale, ckksSchemeParamAttr,
              beforeMulIncludeFirstMul, effectiveScalePolicyKind))) {
        signalPassFailure();
        return;
      }

      SmallVector<mgmt::AdjustScaleOp> candidateNoOpAdjustScales;
      if (failed(validateAdjustScaleResolution(solver,
                                               candidateNoOpAdjustScales))) {
        signalPassFailure();
        return;
      }

      annotateScale(getOperation(), &solver);
      if (failed(runAnnotateMgmt())) {
        signalPassFailure();
        return;
      }

      if (useQAwarePreciseScalePolicy) {
        auto repaired = refinePreciseAdjustScales(getOperation(), materializer);
        if (failed(repaired)) {
          signalPassFailure();
          return;
        }
        if (*repaired) {
          continue;
        }
      }

      noOpAdjustScales = std::move(candidateNoOpAdjustScales);
      refinementConverged = true;
      break;
    }
    if (!refinementConverged) {
      getOperation()->emitOpError()
          << "precise scale refinement exceeded its monotone rewrite bound; "
             "this likely indicates a non-converging adjust_scale repair loop";
      signalPassFailure();
      return;
    }

    for (auto op : noOpAdjustScales) {
      op->replaceAllUsesWith(ValueRange{op.getInput()});
      op->erase();
    }

    DataFlowSolver activeSolver;
    if (failed(createAndRunDataflow(
            getOperation(), activeSolver, logDefaultScale, ckksSchemeParamAttr,
            beforeMulIncludeFirstMul, effectiveScalePolicyKind))) {
      signalPassFailure();
      return;
    }

    bool initScaleFailure = false;
    getOperation()->walk([&](mgmt::InitOp op) {
      auto* lattice = activeSolver.lookupState<ScaleLattice>(op.getResult());
      if (!lattice || lattice->getValue().hasConflict() ||
          !lattice->getValue().isInitialized()) {
        op.emitOpError() << "Dataflow analysis failed to populate scale "
                            "lattice for result\n";
        initScaleFailure = true;
      }
    });
    if (initScaleFailure) {
      signalPassFailure();
      return;
    }

    annotateScale(getOperation(), &activeSolver);
    if (failed(runAnnotateMgmt())) {
      signalPassFailure();
      return;
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Dumping op after annotate-mgmt pass:\n";
      getOperation()->dump();
    });

    LDBG() << "convert adjust_scale to mul_plain";
    CKKSAdjustScaleMaterializer materializer(ckksSchemeParamAttr,
                                             effectiveScalePolicyKind);
    RewritePatternSet patterns(&getContext());
    // TODO(#1641): handle arith.muli in CKKS
    patterns.add<ConvertAdjustScaleToMulPlain<arith::MulFOp>>(&getContext(),
                                                              &materializer);
    (void)walkAndApplyPatterns(getOperation(), std::move(patterns));

    bool materializationFailure = false;
    getOperation()->walk([&](mgmt::AdjustScaleOp op) {
      op.emitOpError() << "failed to materialize scale adjustment";
      materializationFailure = true;
    });
    if (materializationFailure) {
      signalPassFailure();
      return;
    }

    // In precise mode, mgmt.modreduce chains are semantically significant for
    // exact scale tracking and must not be folded into level_reduce forms by
    // generic canonicalization.
    // TODO(#2364): replace this coarse skip with a targeted precise-safe
    // canonicalization strategy once the offending folds are isolated.
    OpPassManager cleanup("builtin.module");
    if (!useQAwarePreciseScalePolicy) {
      cleanup.addPass(createCanonicalizerPass());
    }
    cleanup.addPass(createCSEPass());
    if (failed(runPipeline(cleanup, getOperation()))) {
      signalPassFailure();
      return;
    }

    if (failed(runAnnotateMgmt())) {
      signalPassFailure();
      return;
    }

    DataFlowSolver solverFinal;
    if (failed(createAndRunDataflow(
            getOperation(), solverFinal, logDefaultScale, ckksSchemeParamAttr,
            beforeMulIncludeFirstMul, effectiveScalePolicyKind))) {
      signalPassFailure();
      return;
    }

    annotateScale(getOperation(), &solverFinal);
    if (failed(runAnnotateMgmt())) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace heir
}  // namespace mlir
