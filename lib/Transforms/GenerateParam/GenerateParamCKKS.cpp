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
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Parameters/CKKS/Params.h"
#include "lib/Utils/LogArithmetic.h"
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

// CHEDDAR bootstrap level budget. Unlike lattigo/openfhe -- whose bootstrap
// libraries extend the modulus chain internally at setup time -- CHEDDAR's
// BootContext operates on a single explicit chain that must already contain the
// bootstrap-circuit levels above the compute levels: CoeffToSlot (num_cts) +
// EvalMod (a fixed 8 levels: Log2Ceil(31 mod coeffs)=5 + 3 double-angle) +
// SlotToCoeff (num_stc, which CHEDDAR requires to be >= 2). So when a cheddar
// module bootstraps we grow the generated chain by `kCheddarBootOverhead`
// primes and record the split for `cheddar-configure-crypto-context`. With
// default_encryption_level = computeMaxLevel + num_stc, the boot output lands
// at `default_enc - num_stc = computeMaxLevel`, matching HEIR's level model
// (the bootstrap resets to the compute max), so no level-analysis change is
// needed.
constexpr int kCheddarBootNumCts = 4;
constexpr int kCheddarBootNumStc = 2;
constexpr int kCheddarBootEvalModLevels = 8;
constexpr int kCheddarBootOverhead =
    kCheddarBootNumCts + kCheddarBootNumStc + kCheddarBootEvalModLevels;
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

    std::optional<int> maxLevel = getMaxLevel(getOperation());
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

    // for lattigo (and cheddar, which reuses lattigo's 64-bit CKKS param path),
    // defaults to extended encryption technique
    if (moduleIsLattigo(getOperation()) || moduleIsCheddar(getOperation())) {
      encryptionTechniqueExtended = true;
      LDBG() << "For lattigo/cheddar, fixing extended encryption technique";
    }

    // Lattigo bootstrapping requires LogN >= 14, i.e., ringDim >= 16384.
    // Since ringDim is computed from minSlotCount (minRingDim = 2 *
    // minSlotCount), we bump minSlotCount to 8192 if bootstrapping is
    // present. This is a Lattigo library constraint; cheddar's own bootstrap
    // minimum (256 slots) is enforced by lwe-to-cheddar, and its ring size
    // ends up security-driven by the boot chain's logQP.
    if (moduleIsLattigo(getOperation()) && containsBootstrap(getOperation())) {
      if (minSlotCount < 8192) {
        LDBG() << "Lattigo bootstrapping detected, bumping minSlotCount from "
               << minSlotCount << " to 8192";
        minSlotCount = 8192;
      }
    }

    // CHEDDAR bootstrapping needs the boot-circuit levels physically present in
    // the modulus chain (see kCheddarBootOverhead). Grow the chain accordingly;
    // the compute levels (maxLevel) are unchanged, so level analysis stays
    // valid.
    int computeMaxLevel = maxLevel.value_or(0);
    bool cheddarBoot =
        moduleIsCheddar(getOperation()) && containsBootstrap(getOperation());
    int genMaxLevel =
        cheddarBoot ? computeMaxLevel + kCheddarBootOverhead : computeMaxLevel;

    auto schemeParam = ckks::SchemeParam::getConcreteSchemeParam(
        firstModBits, scalingModBits, genMaxLevel, minSlotCount, usePublicKey,
        encryptionTechniqueExtended, reducedError);

    LDBG() << "Scheme Param:\n" << schemeParam;

    auto* context = &getContext();
    OpBuilder builder(context);
    // Record the cheddar bootstrap split so cheddar-configure-crypto-context
    // can build the BootContext (num_cts/num_stc) and set the Parameter's
    // default_encryption_level consistently with this chain.
    if (cheddarBoot) {
      getOperation()->setAttr("cheddar.boot.num_cts",
                              builder.getI64IntegerAttr(kCheddarBootNumCts));
      getOperation()->setAttr("cheddar.boot.num_stc",
                              builder.getI64IntegerAttr(kCheddarBootNumStc));
      getOperation()->setAttr(
          "cheddar.boot.default_encryption_level",
          builder.getI64IntegerAttr(computeMaxLevel + kCheddarBootNumStc));
    }
    getOperation()->setAttr(kRequestedSlotCountAttrName,
                            builder.getI64IntegerAttr(minSlotCount));
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
