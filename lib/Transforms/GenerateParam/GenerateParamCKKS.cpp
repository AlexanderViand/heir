#include <cmath>
#include <optional>

#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/RangeAnalysis/RangeAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSEnums.h"
#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Parameters/CKKS/Params.h"
#include "lib/Utils/LogArithmetic.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "llvm/include/llvm/Support/DebugLog.h"            // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project
#include "mlir/include/mlir/Support/WalkResult.h"          // from @llvm-project

// IWYU pragma: begin_keep
#include "lib/Transforms/GenerateParam/GenerateParam.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"        // from @llvm-project
// IWYU pragma: end_keep

#define DEBUG_TYPE "generate-param-ckks"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_GENERATEPARAMCKKS
#include "lib/Transforms/GenerateParam/GenerateParam.h.inc"

namespace {
bool containsBootstrap(Operation* op) {
  auto result = op->walk([&](Operation* walkOp) {
    if (isa<ResetsMulDepthOpInterface>(walkOp)) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result.wasInterrupted();
}
}  // namespace

struct GenerateParamCKKS : impl::GenerateParamCKKSBase<GenerateParamCKKS> {
  using GenerateParamCKKSBase::GenerateParamCKKSBase;

  // In CKKS, the modulus for L0 should be larger than the
  // scaling modulus, however, the number of extra bits is often
  // empirically chosen. We use RangeAnalysis to find the
  // maximum number of extra bits needed for the L0 modulus.
  // TODO(#2754): improve this analysis
  std::optional<int> getExtraBitsForLevel0() {
    LDBG() << "Using range analysis to determine extra bits for level 0";
    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    // RangeAnalysis depends on SecretnessAnalysis
    solver.load<SecretnessAnalysis>();
    // For double input in range [-1, 1], we use Log2Arithmetic::of(1) to
    // represent it.
    solver.load<RangeAnalysis>(Log2Arithmetic::of(inputRange));
    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
    }

    std::optional<double> extraBits;

    getOperation()->walk([&](Operation* op) {
      for (auto result : op->getResults()) {
        if (mgmt::shouldHaveMgmtAttribute(result, &solver) &&
            getLevelFromMgmtAttr(result) == 0) {
          auto range = getRange(result, &solver);
          if (range.has_value()) {
            auto resultExtraBits = range->getLog2Value();
            if (!extraBits.has_value() || resultExtraBits > extraBits.value()) {
              extraBits = resultExtraBits;
            }
          }
        }
      }
    });

    if (!extraBits.has_value()) {
      return std::nullopt;
    }
    // 2 more bits for cushion
    int level0ModBits = ceil(extraBits.value()) + 2;
    LDBG() << "Decided on " << level0ModBits << " bits for level 0";
    return level0ModBits;
  }

  void runOnOperation() override {
    LDBG() << "Starting generate-param-ckks pass";
    OpBuilder builder(&getContext());
    getOperation()->setAttr(kCKKSReducedErrorAttrName,
                            builder.getBoolAttr(reducedError));

    if (firstModBits == 0 || validateFirstModBits) {
      auto extraBits = getExtraBitsForLevel0();
      if (!extraBits.has_value()) {
        emitError(getOperation()->getLoc())
            << "Cannot generate CKKS parameters without first modulus bits "
               "or extra bits for level 0.\n";
        signalPassFailure();
        return;
      }

      if (firstModBits == 0) {
        firstModBits = scalingModBits + extraBits.value();
        LDBG() << "First modulus bits not specified, using " << firstModBits
               << " bits.";
      } else if (extraBits.has_value() &&
                 firstModBits - scalingModBits < extraBits.value()) {
        emitWarning(getOperation()->getLoc())
            << "Range Analysis indicate that the first modulus must be larger "
               "than the scaling modulus by at least "
            << extraBits.value() << " bits.\n";
      }
    }
    LDBG() << "First modulus finalized as having " << firstModBits << " bits";

    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    solver.load<SecretnessAnalysis>();
    solver.load<LevelAnalysis>();
    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run level analysis.\n";
      signalPassFailure();
      return;
    }

    std::optional<int> maxLevel = getMaxLevel(getOperation(), &solver);
    LDBG() << "Max level identified as " << maxLevel;

    if (auto schemeParamAttr =
            getOperation()->getAttrOfType<ckks::SchemeParamAttr>(
                ckks::CKKSDialect::kSchemeParamAttrName)) {
      // TODO: put this in validate-noise once CKKS noise model is in
      auto schemeParam = ckks::getSchemeParamFromAttr(schemeParamAttr);
      if (schemeParam.getLevel() < maxLevel.value_or(0)) {
        getOperation()->emitOpError()
            << "The level in the scheme param is smaller than the max level.\n";
        signalPassFailure();
        return;
      }
      return;
    }

    // for lattigo, defaults to extended encryption technique
    if (moduleIsLattigo(getOperation())) {
      encryptionTechniqueExtended = true;
      LDBG() << "For lattigo, fixing extended encryption technique";

      // Lattigo bootstrapping requires LogN >= 14, i.e., ringDim >= 16384.
      // Since ringDim is computed from slotNumber (minRingDim = 2 *
      // slotNumber), we bump slotNumber to 8192 if bootstrapping is present.
      if (containsBootstrap(getOperation())) {
        if (slotNumber < 8192) {
          LDBG() << "Lattigo bootstrapping detected, bumping slotNumber from "
                 << slotNumber << " to 8192";
          slotNumber = 8192;
        }
      }
    }

    // Ensure the chain is long enough for:
    // 1. The target backend's overhead (e.g., OpenFHE's flexible-auto-ext
    //    needs 2 extra levels beyond the circuit's multiplicative depth).
    // 2. Any minimum level hint from the raise pass (preserves noise headroom
    //    from the original Orion parameters), capped at circuitDepth + margin
    //    to avoid over-provisioning simple circuits.
    int effectiveMaxLevel =
        maxLevel.value_or(0) + static_cast<int>(minLevelOverhead);
    if (auto minLevelHint =
            getOperation()->getAttrOfType<IntegerAttr>("heir.min_level_hint")) {
      // Cap the hint at circuitDepth + 3 to avoid over-provisioning simple
      // circuits. The original Orion scheme may have had many more levels
      // than the circuit needs (e.g., 6 primes for a mulDepth=0 circuit).
      int hint = static_cast<int>(minLevelHint.getValue().getSExtValue());
      int cap = maxLevel.value_or(0) + 3;
      effectiveMaxLevel = std::max(effectiveMaxLevel, std::min(hint, cap));
      getOperation()->removeAttr("heir.min_level_hint");
    }
    // Use OpenFHE's security bounds when targeting OpenFHE, to ensure
    // generated parameters are accepted by OpenFHE's runtime validation.
    // OpenFHE's ternary secret key bounds are slightly more generous than
    // the standard HE security guidelines.
    bool useOpenFHEBounds = moduleIsOpenfhe(getOperation());
    auto schemeParam = ckks::SchemeParam::getConcreteSchemeParam(
        firstModBits, scalingModBits, effectiveMaxLevel, slotNumber,
        usePublicKey, encryptionTechniqueExtended, reducedError,
        useOpenFHEBounds);

    LDBG() << "Scheme Param:\n" << schemeParam;

    // If the effective chain is longer than the circuit requires (due to
    // backend overhead or noise headroom hints), record the delta so
    // downstream passes (secret-to-ckks) can shift type levels to match.
    int delta = effectiveMaxLevel - maxLevel.value_or(0);
    if (delta > 0) {
      getOperation()->setAttr("heir.level_offset",
                              builder.getI64IntegerAttr(delta));
    }

    auto* context = &getContext();
    getOperation()->setAttr(kRequestedSlotCountAttrName,
                            builder.getI64IntegerAttr(slotNumber));
    getOperation()->setAttr(
        kActualSlotCountAttrName,
        builder.getI64IntegerAttr(schemeParam.getRingDim() / 2));

    // annotate ckks::SchemeParamAttr to ModuleOp
    getOperation()->setAttr(
        ckks::CKKSDialect::kSchemeParamAttrName,
        ckks::SchemeParamAttr::get(
            context, log2(schemeParam.getRingDim()),
            DenseI64ArrayAttr::get(context, ArrayRef(schemeParam.getQi())),
            DenseI64ArrayAttr::get(context, ArrayRef(schemeParam.getPi())),
            schemeParam.getLogDefaultScale(),
            usePublicKey ? ckks::CKKSEncryptionType::pk
                         : ckks::CKKSEncryptionType::sk,
            encryptionTechniqueExtended
                ? ckks::CKKSEncryptionTechnique::extended
                : ckks::CKKSEncryptionTechnique::standard));
  }
};

}  // namespace heir
}  // namespace mlir
