#include "lib/Dialect/Openfhe/Transforms/ConfigureCryptoContext.h"

#include <cstdint>
#include <limits>
#include <string>

#include "lib/Analysis/MulDepthAnalysis/MulDepthAnalysis.h"
#include "lib/Analysis/RotationAnalysis/RotationAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/BGV/IR/BGVAttributes.h"
#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/BGV/IR/BGVEnums.h"
#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSEnums.h"
#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "lib/Dialect/Openfhe/IR/OpenfheTypes.h"
#include "lib/Dialect/Openfhe/Transforms/ScalingTechniqueUtils.h"
#include "lib/Utils/RotationUtils.h"
#include "lib/Utils/TransformUtils.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "llvm/include/llvm/Support/MathExtras.h"          // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"         // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"       // from @llvm-project
#include "mlir/include/mlir/Support/WalkResult.h"          // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"           // from @llvm-project

#define DEBUG_TYPE "openfhe-configure-crypto-context"

namespace mlir {
namespace heir {
namespace openfhe {

struct Config {
  // computed from IR
  bool hasRelinOp;
  bool hasBootstrapOp;
  bool hasAdvancedSHEOp;
  SmallVector<int64_t> rotIndices;
  // inherited from IR mgmt.openfhe_params
  int64_t evalAddCount;
  int64_t keySwitchCount;
  // inherited from IR LWE type
  int64_t plaintextModulus;
  // inherited from scheme param
  bool encryptionTechniqueExtended;
  // merge from IR and pass options
  int mulDepth;  // user may want to override this; bootstrap also modifies
                 // this
  // from pass options
  // directly applies to GenParamsOp
  int ringDim;
  int batchSize;
  int firstModSize;
  int scalingModSize;
  int digitSize;
  int numLargeDigits;
  int maxRelinSkDeg;
  bool insecure;
  bool keySwitchingTechniqueBV;
  std::string scalingTechnique;
  // for bootstrapping
  int64_t levelBudgetEncode;
  int64_t levelBudgetDecode;
};

namespace {

int inferOpenfheMulDepthFromQSize(size_t qSize, StringRef scalingTechnique) {
  StringRef resolvedScalingTechnique =
      resolveScalingTechnique(scalingTechnique);
  size_t overhead =
      resolvedScalingTechnique == kScalingTechniqueFlexibleAutoExt ? 2 : 1;
  if (qSize < overhead) {
    return 0;
  }
  return static_cast<int>(qSize - overhead);
}

}  // namespace

#define GEN_PASS_DEF_CONFIGURECRYPTOCONTEXT
#include "lib/Dialect/Openfhe/Transforms/Passes.h.inc"

struct ConfigureCryptoContext
    : impl::ConfigureCryptoContextBase<ConfigureCryptoContext> {
  using ConfigureCryptoContextBase::ConfigureCryptoContextBase;

 private:
  Config config;

  // Helper function to check if the function has RelinOp
  bool hasRelinOp(func::FuncOp op) {
    bool result = false;
    walkFuncAndCallees(op, [&](Operation* op) {
      if (isa<openfhe::RelinOp, openfhe::MulOp, openfhe::RelinInPlaceOp>(op)) {
        result = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return result;
  }

  // Helper function to check if the function has BootstrapOp
  bool hasBootstrapOp(func::FuncOp op) {
    bool result = false;
    walkFuncAndCallees(op, [&](Operation* op) {
      if (isa<openfhe::BootstrapOp>(op)) {
        result = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return result;
  }

  bool hasAdvancedSHEOp(func::FuncOp op) {
    bool result = false;
    walkFuncAndCallees(op, [&](Operation* op) {
      if (isa<openfhe::ChebyshevSeriesOp, openfhe::LinearTransformOp,
              openfhe::FastRotationExtOp, openfhe::KeySwitchDownOp>(op)) {
        result = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return result;
  }

  // function that generates the crypto context with proper parameters
  LogicalResult generateGenFunc(func::FuncOp op, const std::string& genFuncName,
                                ImplicitLocOpBuilder& builder) {
    Type openfheContextType =
        openfhe::CryptoContextType::get(builder.getContext());
    SmallVector<Type> funcArgTypes;
    SmallVector<Type> funcResultTypes;
    funcResultTypes.push_back(openfheContextType);

    FunctionType genFuncType =
        FunctionType::get(builder.getContext(), funcArgTypes, funcResultTypes);
    auto genFuncOp = func::FuncOp::create(builder, genFuncName, genFuncType);
    builder.setInsertionPointToEnd(genFuncOp.addEntryBlock());

    Type openfheParamsType = openfhe::CCParamsType::get(builder.getContext());
    Value ccParams = openfhe::GenParamsOp::create(
        builder, openfheParamsType,
        // Essential parameters
        /*mulDepth*/ config.mulDepth,
        /*plainMod*/ config.plaintextModulus,
        // Optional parameters
        /*ringDim*/ config.ringDim,
        /*batchSize*/ config.batchSize,
        /*firstModSize*/ config.firstModSize,
        /*scalingModSize*/ config.scalingModSize,
        /*evalAddCount*/ config.evalAddCount,
        /*keySwitchCount*/ config.keySwitchCount,
        /*digitSize*/ config.digitSize,
        /*numLargeDigits*/ config.numLargeDigits,
        /*maxRelinSkDeg*/ config.maxRelinSkDeg,
        /*insecure*/ config.insecure,
        /*encryptionTechniqueExtended*/ config.encryptionTechniqueExtended,
        /*keySwitchingTechniqueBV*/ config.keySwitchingTechniqueBV,
        /*scalingTechnique*/ builder.getStringAttr(config.scalingTechnique));
    Value cryptoContext = openfhe::GenContextOp::create(
        builder, openfheContextType, ccParams,
        BoolAttr::get(builder.getContext(),
                      config.hasBootstrapOp || config.hasAdvancedSHEOp));

    func::ReturnOp::create(builder, cryptoContext);
    return success();
  }

  // function that configures the crypto context with proper keygeneration
  LogicalResult generateConfigFunc(func::FuncOp op,
                                   const std::string& configFuncName,
                                   ImplicitLocOpBuilder& builder) {
    Type openfheContextType =
        openfhe::CryptoContextType::get(builder.getContext());
    Type privateKeyType = openfhe::PrivateKeyType::get(builder.getContext());

    SmallVector<Type> funcArgTypes;
    funcArgTypes.push_back(openfheContextType);
    funcArgTypes.push_back(privateKeyType);

    SmallVector<Type> funcResultTypes;
    funcResultTypes.push_back(openfheContextType);

    FunctionType configFuncType =
        FunctionType::get(builder.getContext(), funcArgTypes, funcResultTypes);
    auto configFuncOp =
        func::FuncOp::create(builder, configFuncName, configFuncType);
    builder.setInsertionPointToEnd(configFuncOp.addEntryBlock());

    Value cryptoContext = configFuncOp.getArgument(0);
    Value privateKey = configFuncOp.getArgument(1);

    if (config.hasRelinOp || config.hasBootstrapOp || config.hasAdvancedSHEOp) {
      openfhe::GenMulKeyOp::create(builder, cryptoContext, privateKey);
    }
    if (!config.rotIndices.empty()) {
      openfhe::GenRotKeyOp::create(builder, cryptoContext, privateKey,
                                   config.rotIndices);
    }
    if (config.hasBootstrapOp) {
      openfhe::SetupBootstrapOp::create(
          builder, cryptoContext,
          IntegerAttr::get(IndexType::get(builder.getContext()),
                           config.levelBudgetEncode),
          IntegerAttr::get(IndexType::get(builder.getContext()),
                           config.levelBudgetDecode));
      openfhe::GenBootstrapKeyOp::create(builder, cryptoContext, privateKey);
    }

    func::ReturnOp::create(builder, cryptoContext);
    return success();
  }

  LogicalResult convertFunc(func::FuncOp op) {
    auto module = op->getParentOfType<ModuleOp>();
    std::string genFuncName("");
    llvm::raw_string_ostream genNameOs(genFuncName);
    genNameOs << op.getSymName() << "__generate_crypto_context";

    std::string configFuncName("");
    llvm::raw_string_ostream configNameOs(configFuncName);
    configNameOs << op.getSymName() << "__configure_crypto_context";

    ImplicitLocOpBuilder builder =
        ImplicitLocOpBuilder::atBlockEnd(module.getLoc(), module.getBody());

    if (failed(generateGenFunc(op, genFuncName, builder))) {
      return failure();
    }

    builder.setInsertionPointToEnd(module.getBody());

    if (failed(generateConfigFunc(op, configFuncName, builder))) {
      return failure();
    }
    return success();
  }

  LogicalResult getConfig(func::FuncOp op) {
    auto module = op->getParentOfType<ModuleOp>();
    int inferredRingDim = 0;
    int inferredBatchSize = 0;
    int inferredFirstModSize = 0;
    int inferredScalingModSize = 0;
    int inferredMulDepth = 0;
    std::string inferredScalingTechnique;
    bool inferredCkksSchemeParam = false;

    if (auto scalingTechniqueAttr =
            module->getAttrOfType<StringAttr>(kScalingTechniqueAttrName)) {
      inferredScalingTechnique = scalingTechniqueAttr.getValue().str();
    }
    if (auto requestedSlotCountAttr =
            module->getAttrOfType<IntegerAttr>(kRequestedSlotCountAttrName)) {
      inferredBatchSize =
          static_cast<int>(requestedSlotCountAttr.getValue().getSExtValue());
    }

    // remove bgv.schemeParam attribute if present
    // fill encryptionTechniqueExtended and plaintextModulus
    config.encryptionTechniqueExtended = false;
    config.plaintextModulus = 0;
    if (auto schemeParamAttr = module->getAttrOfType<bgv::SchemeParamAttr>(
            bgv::BGVDialect::kSchemeParamAttrName)) {
      if (moduleIsBGV(module) && schemeParamAttr.getEncryptionTechnique() ==
                                     bgv::BGVEncryptionTechnique::extended) {
        module->emitError(
            "Extended encryption technique is not supported in OpenFHE BGV");
        return failure();
      }
      config.encryptionTechniqueExtended =
          schemeParamAttr.getEncryptionTechnique() ==
          bgv::BGVEncryptionTechnique::extended;
      if (!moduleIsCKKS(module))
        config.plaintextModulus = schemeParamAttr.getPlaintextModulus();
      module->removeAttr(bgv::BGVDialect::kSchemeParamAttrName);
    }

    // remove ckks.schemeParam attribute if present
    // For CKKS, plainMod must be 0 to avoid codegen for SetPlaintextModulus.
    // OpenFHE will throw an exception if you try to set the plaintext modulus
    // in CKKS.
    if (auto schemeParamAttr = module->getAttrOfType<ckks::SchemeParamAttr>(
            ckks::CKKSDialect::kSchemeParamAttrName)) {
      inferredCkksSchemeParam = true;
      if (schemeParamAttr.getEncryptionTechnique() ==
          ckks::CKKSEncryptionTechnique::extended) {
        module->emitError(
            "Extended encryption technique is not supported in OpenFHE CKKS");
        return failure();
      }
      inferredRingDim =
          static_cast<int>(int64_t{1} << schemeParamAttr.getLogN());
      if (inferredBatchSize == 0) {
        inferredBatchSize =
            static_cast<int>((int64_t{1} << schemeParamAttr.getLogN()) / 2);
      }
      auto q = schemeParamAttr.getQ();
      if (!q.empty()) {
        // CKKS schemeParam stores the actual Q chain length chosen by
        // management. OpenFHE's multiplicative-depth parameter is one smaller
        // than |Q| in the ordinary modes and two smaller in FLEXIBLEAUTOEXT.
        StringRef effectiveScalingTechnique =
            scalingTechnique.empty() ? StringRef(inferredScalingTechnique)
                                     : StringRef(scalingTechnique);
        inferredMulDepth =
            inferOpenfheMulDepthFromQSize(q.size(), effectiveScalingTechnique);
        auto firstQ = static_cast<uint64_t>(q[0]);
        inferredFirstModSize =
            std::numeric_limits<uint64_t>::digits - llvm::countl_zero(firstQ);
      }
      inferredScalingModSize = schemeParamAttr.getLogDefaultScale();
      module->removeAttr(ckks::CKKSDialect::kSchemeParamAttrName);
    }

    config.mulDepth = 0;
    if (!moduleIsCKKS(module) || !inferredCkksSchemeParam) {
      LLVM_DEBUG(llvm::dbgs() << "Recomputing mul depth\n");
      DataFlowSolver solver;
      dataflow::loadBaselineAnalyses(solver);
      solver.load<SecretnessAnalysis>();
      solver.load<MulDepthAnalysis>();

      if (failed(solver.initializeAndRun(module))) {
        op->emitOpError() << "Failed to run mul depth analysis.\n";
        return failure();
      }

      walkValues(op, [&](Value value) {
        auto mulDepthState =
            solver.lookupState<MulDepthLattice>(value)->getValue();
        if (!mulDepthState.isInitialized()) {
          LLVM_DEBUG(llvm::dbgs()
                     << "mul depth uninitialized at " << value << "\n");
          return;
        }
        auto mulDepth = mulDepthState.getMulDepth();
        if (mulDepth > config.mulDepth) {
          LLVM_DEBUG(llvm::dbgs()
                     << "Found larger mul depth=" << mulDepth << "\n");
          config.mulDepth = mulDepth;
        }
      });
    }

    config.hasBootstrapOp = hasBootstrapOp(op);
    config.hasAdvancedSHEOp = hasAdvancedSHEOp(op);

    // pass option could override mulDepth
    if (mulDepth != 0) {
      config.mulDepth = mulDepth;
    } else if (inferredMulDepth != 0) {
      config.mulDepth = inferredMulDepth;
    }

    if (moduleIsCKKS(module) && !inferredCkksSchemeParam && mulDepth == 0 &&
        (config.hasBootstrapOp || config.hasAdvancedSHEOp)) {
      op.emitOpError()
          << "CKKS OpenFHE configuration for bootstrap, linear_transform, "
             "and chebyshev_series requires a precomputed `ckks.schemeParam` "
             "or an explicit `--openfhe-configure-crypto-context=mul-depth=` "
             "override";
      return failure();
    }

    // relin and rotation
    config.hasRelinOp = hasRelinOp(op);

    LLVM_DEBUG(llvm::dbgs() << "Starting rotation analysis\n");
    RotationAnalysis analysis;
    LogicalResult result = analysis.run(module);
    if (failed(result)) {
      return op->emitOpError("failed to compute static rotation indices");
    }

    auto setIndices = analysis.getRotationIndices();
    module.walk([&](openfhe::LinearTransformOp ltOp) {
      auto diagonalsType =
          dyn_cast<RankedTensorType>(ltOp.getDiagonals().getType());
      if (!diagonalsType || diagonalsType.getRank() != 2) {
        return;
      }
      int64_t slots = diagonalsType.getShape()[1];
      auto ltIndices = lintransRotationIndices(
          ltOp.getDiagonalIndicesAttr().asArrayRef(), slots,
          ltOp.getLogBabyStepGiantStepRatioAttr().getInt());
      setIndices.insert(ltIndices.begin(), ltIndices.end());
    });
    SmallVector<int64_t> indices(setIndices.begin(), setIndices.end());
    LLVM_DEBUG({
      llvm::dbgs() << "Finished rotation analysis; found " << indices.size()
                   << " rotations which were=\n";
      for (int64_t shift : indices) {
        llvm::dbgs() << shift << ", ";
      }
      llvm::dbgs() << "\n";
    });
    config.rotIndices = indices;

    // get evalAddCount/KeySwitchCount from func attribute, if present
    config.evalAddCount = 0;
    config.keySwitchCount = 0;
    if (auto openfheParamsAttr = op->getAttrOfType<mgmt::OpenfheParamsAttr>(
            mgmt::MgmtDialect::kArgOpenfheParamsAttrName)) {
      config.evalAddCount = openfheParamsAttr.getEvalAddCount();
      config.keySwitchCount = openfheParamsAttr.getKeySwitchCount();
      // remove the attribute after reading
      op->removeAttr(mgmt::MgmtDialect::kArgOpenfheParamsAttrName);
    }

    // fill config with pass options
    config.ringDim = ringDim != 0 ? ringDim : inferredRingDim;
    if (moduleIsCKKS(module)) {
      config.batchSize = batchSize != 0 ? batchSize : inferredBatchSize;
    } else {
      config.batchSize = batchSize;
    }
    config.firstModSize =
        firstModSize != 0 ? firstModSize : inferredFirstModSize;
    config.scalingModSize =
        scalingModSize != 0 ? scalingModSize : inferredScalingModSize;
    config.digitSize = digitSize;
    config.numLargeDigits = numLargeDigits;
    config.maxRelinSkDeg = maxRelinSkDeg;
    config.insecure = insecure;
    config.keySwitchingTechniqueBV = keySwitchingTechniqueBV;
    if (!isSupportedScalingTechnique(scalingTechnique)) {
      module->emitError() << "unsupported OpenFHE scaling technique `"
                          << scalingTechnique << "`";
      return failure();
    }
    config.scalingTechnique =
        scalingTechnique.empty() ? inferredScalingTechnique : scalingTechnique;
    if (moduleIsCKKS(module) && inferredCkksSchemeParam &&
        config.scalingTechnique.empty()) {
      config.scalingTechnique = kScalingTechniqueFixedManual.str();
      module->emitRemark()
          << "No OpenFHE scaling technique specified for CKKS; "
             "defaulting to fixed-manual. Pass "
             "`--openfhe-configure-crypto-context=scaling-technique=...` "
             "or use the OpenFHE-aware CKKS management pipeline to set it.";
    }
    config.levelBudgetDecode = levelBudgetDecode;
    config.levelBudgetEncode = levelBudgetEncode;

    // for BFV, keep only one of MulDepth/EvalAddCount/KeySwitchCount
    // If MulDepth != 0, clean EvalAddCount/KeySwitchCount
    // If MulDepth == 0, and both Count are non-zero, emit warning
    //
    // https://github.com/openfheorg/openfhe-development/blob/v1.3.0/src/pke/lib/scheme/bfvrns/bfvrns-parametergeneration.cpp
    if (moduleIsBFV(module)) {
      if (config.mulDepth != 0) {
        config.evalAddCount = 0;
        config.keySwitchCount = 0;
      } else if (config.evalAddCount != 0 && config.keySwitchCount != 0) {
        module->emitWarning(
            "MulDepth is 0, but EvalAddCount and KeySwitchCount are both "
            "non-zero. This may not satisfy the correctness requirement of "
            "OpenFHE BFV Parameter Generation");
      }
    }
    return success();
  }

 public:
  void runOnOperation() override {
    // The rotation analysis requires -sccp is run first to propagate constants
    // as much as possible.
    OpPassManager pipeline("builtin.module");
    pipeline.addPass(createSCCPPass());
    pipeline.addPass(createCanonicalizerPass());
    (void)runPipeline(pipeline, getOperation());

    auto funcOp =
        detectEntryFunction(cast<ModuleOp>(getOperation()), entryFunction);
    if (!funcOp) {
      return;
    }
    if (failed(getConfig(funcOp))) {
      signalPassFailure();
      return;
    }
    if (failed(convertFunc(funcOp))) {
      funcOp->emitError("Failed to configure the crypto context for func");
      signalPassFailure();
    }
  }
};

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
