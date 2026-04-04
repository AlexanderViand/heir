#include <utility>

#include "lib/Analysis/ScaleAnalysis/ScaleAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
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

class CKKSAdjustScaleMaterializer : public AdjustScaleMaterializer {
 public:
  virtual ~CKKSAdjustScaleMaterializer() = default;

  FailureOr<APInt> deltaScale(mgmt::AdjustScaleOp op, const APInt& scale,
                              const APInt& inputScale) const override {
    return divideUnsignedAPIntExact(scale, inputScale);
  }
};

#define GEN_PASS_DEF_POPULATESCALECKKS
#include "lib/Transforms/PopulateScale/PopulateScale.h.inc"

LogicalResult createAndRunDataflow(Operation* op, DataFlowSolver& solver,
                                   int64_t logDefaultScale,
                                   ckks::SchemeParamAttr ckksSchemeParamAttr,
                                   bool beforeMulIncludeFirstMul) {
  SymbolTableCollection symbolTable;
  dataflow::loadBaselineAnalyses(solver);
  solver.load<SecretnessAnalysis>();
  APInt inputScale = getNominalPowerOfTwoScaleFromLog2(logDefaultScale);
  if (beforeMulIncludeFirstMul) {
    LDBG() << "Encoding at scale^2 due to 'include-first-mul' config";
    inputScale = multiplyUnsignedAPIntExact(inputScale, inputScale);
  }
  auto param = ckks::getSchemeParamFromAttr(ckksSchemeParamAttr);
  solver.load<ScaleAnalysis<CKKSScaleModel>>(param,
                                             /*inputScale*/ inputScale);
  // Back-prop ScaleAnalysis depends on (forward) ScaleAnalysis
  solver.load<ScaleAnalysisBackward<CKKSScaleModel>>(symbolTable, param);

  return solver.initializeAndRun(op);
}

struct PopulateScaleCKKS : impl::PopulateScaleCKKSBase<PopulateScaleCKKS> {
  using PopulateScaleCKKSBase::PopulateScaleCKKSBase;

  void runOnOperation() override {
    auto ckksSchemeParamAttr = mlir::dyn_cast<ckks::SchemeParamAttr>(
        getOperation()->getAttr(ckks::CKKSDialect::kSchemeParamAttrName));
    auto logDefaultScale = ckksSchemeParamAttr.getLogDefaultScale();

    OpPassManager normalizeMgmt("builtin.module");
    normalizeMgmt.addPass(mgmt::createAnnotateMgmt());
    if (failed(runPipeline(normalizeMgmt, getOperation()))) {
      signalPassFailure();
      return;
    }

    DataFlowSolver solver;
    if (failed(createAndRunDataflow(getOperation(), solver, logDefaultScale,
                                    ckksSchemeParamAttr,
                                    beforeMulIncludeFirstMul))) {
      signalPassFailure();
      return;
    }

    // At this point all adjust_scale and mgmt.init results must have fully
    // resolved nominal scales. Leaving them unresolved means the pre-lowering
    // symbolic management structure is inconsistent with the current policy.
    bool scaleConflict = false;
    SmallVector<mgmt::AdjustScaleOp> noOpAdjustScales;
    getOperation()->walk([&](mgmt::AdjustScaleOp op) {
      auto* lattice = solver.lookupState<ScaleLattice>(op.getResult());
      if (!lattice) {
        op.emitOpError() << "Dataflow analysis failed to populate scale "
                            "lattice for result\n";
        scaleConflict = true;
        return WalkResult::interrupt();
      }
      if (lattice->getValue().hasConflict()) {
        op.emitOpError() << "Dataflow analysis found conflicting exact scales "
                            "for result\n";
        scaleConflict = true;
        return WalkResult::interrupt();
      }
      if (!lattice->getValue().isInitialized()) {
        auto* inputLattice = solver.lookupState<ScaleLattice>(op.getInput());
        if (inputLattice && inputLattice->getValue().isInitialized() &&
            !inputLattice->getValue().hasConflict()) {
          // If nothing downstream constrains the target scale, the symbolic
          // adjust_scale is a no-op and can be erased before materialization.
          noOpAdjustScales.push_back(op);
          return WalkResult::advance();
        }
        op.emitOpError()
            << "Dataflow analysis did not resolve a nominal target scale\n";
        scaleConflict = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (scaleConflict) {
      signalPassFailure();
      return;
    }

    for (auto op : noOpAdjustScales) {
      op->replaceAllUsesWith(ValueRange{op.getInput()});
      op->erase();
    }

    DataFlowSolver* activeSolver = &solver;
    DataFlowSolver refinedSolver;
    if (!noOpAdjustScales.empty()) {
      if (failed(createAndRunDataflow(getOperation(), refinedSolver,
                                      logDefaultScale, ckksSchemeParamAttr,
                                      beforeMulIncludeFirstMul))) {
        signalPassFailure();
        return;
      }
      activeSolver = &refinedSolver;
    }

    bool initScaleFailure = false;
    getOperation()->walk([&](mgmt::InitOp op) {
      auto* lattice = activeSolver->lookupState<ScaleLattice>(op.getResult());
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

    annotateScale(getOperation(), activeSolver);
    OpPassManager annotateMgmt("builtin.module");
    annotateMgmt.addPass(mgmt::createAnnotateMgmt());
    if (failed(runPipeline(annotateMgmt, getOperation()))) {
      signalPassFailure();
      return;
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Dumping op after annotate-mgmt pass:\n";
      getOperation()->dump();
    });

    LDBG() << "convert adjust_scale to mul_plain";
    RewritePatternSet patterns(&getContext());
    CKKSAdjustScaleMaterializer materializer;
    // TODO(#1641): handle arith.muli in CKKS
    patterns.add<ConvertAdjustScaleToMulPlain<arith::MulFOp>>(&getContext(),
                                                              &materializer);
    (void)walkAndApplyPatterns(getOperation(), std::move(patterns));

    bool materializationFailure = false;
    getOperation()->walk([&](mgmt::AdjustScaleOp op) {
      op.emitOpError() << "failed to materialize nominal scale adjustment";
      materializationFailure = true;
    });
    if (materializationFailure) {
      signalPassFailure();
      return;
    }

    // run canonicalizer and CSE to clean up arith.constant and move no-op out
    // of the secret.generic
    OpPassManager cleanup("builtin.module");
    cleanup.addPass(createCanonicalizerPass());
    cleanup.addPass(createCSEPass());
    if (failed(runPipeline(cleanup, getOperation()))) {
      signalPassFailure();
      return;
    }

    DataFlowSolver solverFinal;
    if (failed(createAndRunDataflow(getOperation(), solverFinal,
                                    logDefaultScale, ckksSchemeParamAttr,
                                    beforeMulIncludeFirstMul))) {
      signalPassFailure();
      return;
    }

    annotateScale(getOperation(), &solverFinal);
    OpPassManager annotateFinalMgmt("builtin.module");
    annotateFinalMgmt.addPass(mgmt::createAnnotateMgmt());
    if (failed(runPipeline(annotateFinalMgmt, getOperation()))) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace heir
}  // namespace mlir
