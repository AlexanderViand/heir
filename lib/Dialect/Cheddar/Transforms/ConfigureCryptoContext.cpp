#include "lib/Dialect/Cheddar/Transforms/ConfigureCryptoContext.h"

#include <algorithm>
#include <string>
#include <utility>

#include "lib/Analysis/RotationAnalysis/RotationAnalysis.h"
#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/Cheddar/IR/CheddarOps.h"
#include "lib/Dialect/Cheddar/IR/CheddarTypes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Utils/TransformUtils.h"
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"          // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"         // from @llvm-project

namespace mlir::heir::cheddar {

#define GEN_PASS_DEF_CHEDDARCONFIGURECRYPTOCONTEXT
#include "lib/Dialect/Cheddar/Transforms/ConfigureCryptoContext.h.inc"

namespace {

// scale-snu CHEDDAR's pinned BootParameter consumes eight levels in EvalMod:
// ceil(log2(31 minimax coefficients)) + three double-angle steps. Its
// BootContext requires Parameter::default_encryption_level_ to equal the
// resulting SlotToCoeff start level.
constexpr int64_t kBootstrapEvalModLevels = 8;
constexpr int64_t kMinBootstrapSlots = 256;

// Build setup and key-generation functions in destination-passing tensor form:
//   %p   = cheddar.make_parameter ...
//   %ctx = cheddar.create_context %p, %ctx_init
//   return %ctx
//
//   %ui0 = cheddar.create_user_interface %ctx, %ui_init
//   %ui1 = cheddar.prepare_rot_key %ui0 {distance, maxLevel}   // per distance
//   return %ctx, %uiN
// Params come from the CKKS scheme attrs (logN/scale/Q/P); the rotation
// distances come from the shared RotationAnalysis (the same way the lattigo and
// openfhe backends discover which keys to generate). The two tensor results
// become owning context/user_interface out-params during Cheddar
// bufferization, and the cheddar-to-emitc boundary re-types them to owning
// smart-pointer references.
// When the program bootstraps, the generated configure builds a BootContext
// (and runs the one-time bootstrap precompute + rotation-key request) instead
// of a plain Context: the entry then takes a `!cheddar.boot_context`.
// `bootstrapNumSlots` is the number of slots used to prepare the bootstrap;
// `numCtsLevels` / `numStcLevels` are the CtS/StC level budgets (a
// depth/rotations trade-off, cf. OpenFHE's level-budget-encode/decode),
// threaded in as pass options.
void buildConfigureFuncs(ModuleOp moduleOp, func::FuncOp entry, int64_t logN,
                         int64_t logScale, DenseI64ArrayAttr Q,
                         DenseI64ArrayAttr P,
                         ArrayRef<std::pair<int64_t, int64_t>> rotationKeys,
                         bool bootstraps, int64_t bootstrapNumSlots,
                         int64_t numCtsLevels, int64_t numStcLevels,
                         int64_t defaultEncLevel, int64_t denseHammingWeight,
                         int64_t sparseHammingWeight, int64_t logMessageRatio) {
  MLIRContext* ctx = moduleOp.getContext();
  // EvalMod message headroom passed to CHEDDAR's BootParameter. This is the
  // reserved bits for the MESSAGE magnitude (~log2(max|m|)+margin), NOT a
  // function of the modulus chain: CHEDDAR scales the level-0 message UP by
  // (log2(q0/scale) - log_message_ratio) bits before EvalMod, so the message
  // fills the accurate part of the fixed sine (sin 2*pi*x) minimax. Too LARGE a
  // ratio under-scales the message into the inaccurate low end of the sine ->
  // the bootstrap stops being identity and silently corrupts its output (which
  // then detonates a downstream Chebyshev eval). The old
  // firstModBits-logScale-2 formula tied the headroom to the chain and gave ~13
  // (log_scaleup_ ~= 2, a 256x under-scale). For the normalized activations
  // these programs bootstrap
  // (|m| ~ O(1), the sign/ReLU inputs), CHEDDAR's default headroom of 5 is the
  // right magnitude (log_scaleup_ ~= 10). Override via the `log-message-ratio`
  // option for a program with a larger message bound.
  int64_t effLogMessageRatio = logMessageRatio;
  if (bootstraps && effLogMessageRatio < 0) {
    effLogMessageRatio = 5;
  }

  OpBuilder builder(ctx);
  builder.setInsertionPointToEnd(moduleOp.getBody());
  Location loc = entry.getLoc();

  // Bootstrapping programs hand back an owning BootContext; others a Context.
  Type ctxElt = bootstraps ? Type(BootContextType::get(ctx))
                           : Type(ContextType::get(ctx));
  auto ctxTensor = RankedTensorType::get({}, ctxElt);
  auto uiTensor = RankedTensorType::get({}, UserInterfaceType::get(ctx));
  auto roleAttr = builder.getDictionaryAttr({builder.getNamedAttr(
      kClientHelperFuncName, builder.getStringAttr(entry.getSymName()))});

  std::string setupName = (entry.getSymName() + "__setup").str();
  auto setupType = FunctionType::get(ctx, {}, {ctxTensor});
  auto setupFunc = func::FuncOp::create(builder, loc, setupName, setupType);
  setupFunc.setPublic();
  setupFunc->setAttr(kClientSetupFuncAttrName, roleAttr);

  Block* bodyBlock = setupFunc.addEntryBlock();
  builder.setInsertionPointToStart(bodyBlock);

  auto i64 = [&](int64_t v) { return builder.getI64IntegerAttr(v); };
  // Bootstrapping pins default_encryption_level below the chain top and sets
  // the secret-key hamming weights; non-boot leaves these null (emitter
  // defaults).
  IntegerAttr defaultEncAttr =
      bootstraps ? i64(defaultEncLevel) : IntegerAttr();
  IntegerAttr denseHwAttr =
      bootstraps ? i64(denseHammingWeight) : IntegerAttr();
  IntegerAttr sparseHwAttr =
      bootstraps ? i64(sparseHammingWeight) : IntegerAttr();
  Value params =
      MakeParameterOp::create(builder, loc, ParameterType::get(ctx), i64(logN),
                              i64(logScale), Q, P, defaultEncAttr, denseHwAttr,
                              sparseHwAttr)
          .getParams();
  // Shape-only DPS destinations. Cheddar bufferization connects these to the
  // caller-provided result destinations before One-Shot Bufferize.
  Value ctxInit = tensor::EmptyOp::create(builder, loc, ctxTensor.getShape(),
                                          ctxTensor.getElementType());
  Value context =
      bootstraps
          ? CreateBootContextOp::create(
                builder, loc, TypeRange{ctxTensor}, params, i64(numCtsLevels),
                i64(numStcLevels), i64(effLogMessageRatio), ctxInit)
                ->getResult(0)
          : CreateContextOp::create(builder, loc, TypeRange{ctxTensor},
                                    ValueRange{params, ctxInit})
                ->getResult(0);
  func::ReturnOp::create(builder, loc, context);

  builder.setInsertionPointToEnd(moduleOp.getBody());
  std::string keygenName = (entry.getSymName() + "__keygen").str();
  auto keygenType = FunctionType::get(ctx, {ctxTensor}, {ctxTensor, uiTensor});
  auto keygenFunc = func::FuncOp::create(builder, loc, keygenName, keygenType);
  keygenFunc.setPublic();
  keygenFunc->setAttr(kClientKeygenFuncAttrName, roleAttr);
  bodyBlock = keygenFunc.addEntryBlock();
  builder.setInsertionPointToStart(bodyBlock);
  context = keygenFunc.getArgument(0);
  Value uiInit = tensor::EmptyOp::create(builder, loc, uiTensor.getShape(),
                                         uiTensor.getElementType());
  Value ui = CreateUserInterfaceOp::create(builder, loc, TypeRange{uiTensor},
                                           ValueRange{context, uiInit})
                 ->getResult(0);
  for (auto [distance, level] : rotationKeys)
    ui = PrepareRotKeyOp::create(builder, loc, TypeRange{uiTensor}, ui,
                                 i64(distance), i64(level))
             ->getResult(0);
  if (bootstraps) {
    auto prepare =
        PrepareBootstrapOp::create(builder, loc, TypeRange{ctxTensor, uiTensor},
                                   context, ui, i64(bootstrapNumSlots));
    context = prepare->getResult(0);
    ui = prepare->getResult(1);
  }
  func::ReturnOp::create(builder, loc, ValueRange{context, ui});

  // Keep the combined setup and key-generation entry point.
  builder.setInsertionPointToEnd(moduleOp.getBody());
  auto configureType = FunctionType::get(ctx, {}, {ctxTensor, uiTensor});
  std::string configureName = (entry.getSymName() + "__configure").str();
  auto configureFunc =
      func::FuncOp::create(builder, loc, configureName, configureType);
  configureFunc.setPublic();
  bodyBlock = configureFunc.addEntryBlock();
  builder.setInsertionPointToStart(bodyBlock);
  auto setupCall = func::CallOp::create(builder, loc, setupFunc, ValueRange{});
  auto keygenCall =
      func::CallOp::create(builder, loc, keygenFunc, setupCall.getResult(0));
  func::ReturnOp::create(builder, loc, keygenCall.getResults());
}

}  // namespace

struct CheddarConfigureCryptoContext
    : public impl::CheddarConfigureCryptoContextBase<
          CheddarConfigureCryptoContext> {
  using CheddarConfigureCryptoContextBase::CheddarConfigureCryptoContextBase;

  void runOnOperation() override {
    auto moduleOp = cast<ModuleOp>(getOperation());
    MLIRContext* ctx = &getContext();

    // RotationAnalysis requires -sccp to have propagated constants so the
    // rotation indices are statically detectable (mirrors the lattigo/openfhe
    // configure passes).
    OpPassManager pipeline("builtin.module");
    pipeline.addPass(createSCCPPass());
    pipeline.addPass(createCanonicalizerPass());
    if (failed(runPipeline(pipeline, moduleOp))) {
      signalPassFailure();
      return;
    }

    auto schemeParamAttr = moduleOp->getAttrOfType<ckks::SchemeParamAttr>(
        ckks::CKKSDialect::kSchemeParamAttrName);
    if (!schemeParamAttr) return;

    DenseI64ArrayAttr Q = schemeParamAttr.getQ();
    DenseI64ArrayAttr P = schemeParamAttr.getP();
    if (!Q || Q.size() == 0 || !P || P.size() == 0) {
      moduleOp.emitError(
          "CHEDDAR context configuration requires non-empty CKKS Q and P "
          "modulus chains");
      signalPassFailure();
      return;
    }

    auto entry = detectEntryFunction(moduleOp, entryFunction);
    if (!entry) {
      signalPassFailure();
      return;
    }
    for (StringRef suffix : {"__setup", "__keygen", "__configure"}) {
      std::string functionName = (entry.getSymName() + suffix).str();
      if (moduleOp.lookupSymbol<func::FuncOp>(functionName)) {
        entry.emitOpError()
            << "configuration function @" << functionName << " already exists";
        signalPassFailure();
        return;
      }
    }

    RotationAnalysis rotationAnalysis;
    if (failed(rotationAnalysis.run(moduleOp))) {
      entry.emitOpError("failed to compute static rotation indices");
      signalPassFailure();
      return;
    }
    bool bootstraps = false;
    moduleOp.walk([&](BootOp) { bootstraps = true; });
    int64_t bootstrapNumSlots = 0;
    if (bootstraps) {
      auto slotsAttr =
          moduleOp->getAttrOfType<IntegerAttr>(kRequestedSlotCountAttrName);
      if (!slotsAttr) {
        entry.emitOpError(
            "bootstrapping program is missing the scheme.requested_slot_count "
            "module attribute needed to configure the boot context");
        signalPassFailure();
        return;
      }
      // EvalSpecialFFT requires at least 256 slots. BootContext supports
      // bootstrapping a smaller ciphertext with a larger prepared transform.
      bootstrapNumSlots = std::max(slotsAttr.getInt(), kMinBootstrapSlots);
    }

    int64_t logN = schemeParamAttr.getLogN();
    int64_t logDefaultScale = schemeParamAttr.getLogDefaultScale();
    int64_t bootNumCts = numCtsLevels;
    int64_t bootNumStc = numStcLevels;
    int64_t defaultEncLevel = static_cast<int64_t>(Q.size()) - 1;
    int64_t denseHammingWeight = 0;
    int64_t sparseHammingWeight = 0;
    if (bootstraps) {
      if (auto attr =
              moduleOp->getAttrOfType<IntegerAttr>("cheddar.boot.num_cts"))
        bootNumCts = attr.getInt();
      if (auto attr =
              moduleOp->getAttrOfType<IntegerAttr>("cheddar.boot.num_stc"))
        bootNumStc = attr.getInt();
      if (bootNumCts < 0 || bootNumStc < 0) {
        entry.emitOpError(
            "CHEDDAR bootstrap CtS and StC level counts must be non-negative");
        signalPassFailure();
        return;
      }
      defaultEncLevel -= bootNumCts + kBootstrapEvalModLevels;
      int64_t bootstrapEndLevel = defaultEncLevel - bootNumStc;
      if (defaultEncLevel < 0 || bootstrapEndLevel < 0) {
        entry.emitOpError()
            << "CHEDDAR bootstrap modulus chain is too short: " << Q.size()
            << " Q primes cannot cover " << bootNumCts << " CtS + "
            << kBootstrapEvalModLevels << " EvalMod + " << bootNumStc
            << " StC levels";
        signalPassFailure();
        return;
      }
      denseHammingWeight = int64_t{1} << (logN - 1);
      sparseHammingWeight = 32;
    }

    SmallVector<std::pair<int64_t, int64_t>> rotationKeys;
    if (prepareRotationKeysAtUseLevels) {
      WalkResult result = moduleOp.walk([&](RotationOpInterface rotationOp) {
        auto level = rotationOp->getAttrOfType<IntegerAttr>("level");
        if (!level) {
          rotationOp.emitOpError(
              "requires a level attribute when preparing rotation keys at "
              "operation use levels");
          return WalkResult::interrupt();
        }
        for (int64_t distance :
             rotationAnalysis.getRotationIndices(rotationOp)) {
          rotationKeys.emplace_back(distance, level.getInt());
        }
        return WalkResult::advance();
      });
      if (result.wasInterrupted()) {
        signalPassFailure();
        return;
      }
    } else {
      int64_t rotationKeyLevel =
          bootstraps ? defaultEncLevel : static_cast<int64_t>(Q.size()) - 1;
      for (int64_t distance : rotationAnalysis.getRotationIndices())
        rotationKeys.emplace_back(distance, rotationKeyLevel);
    }
    llvm::sort(rotationKeys, [](const auto& lhs, const auto& rhs) {
      if (lhs.first != rhs.first) return lhs.first < rhs.first;
      return lhs.second > rhs.second;
    });
    rotationKeys.erase(std::unique(rotationKeys.begin(), rotationKeys.end()),
                       rotationKeys.end());

    moduleOp->setAttr("cheddar.logN",
                      IntegerAttr::get(IntegerType::get(ctx, 64), logN));
    moduleOp->setAttr(
        "cheddar.logDefaultScale",
        IntegerAttr::get(IntegerType::get(ctx, 64), logDefaultScale));
    moduleOp->setAttr("cheddar.Q", Q);
    moduleOp->setAttr("cheddar.P", P);
    buildConfigureFuncs(moduleOp, entry, logN, logDefaultScale, Q, P,
                        rotationKeys, bootstraps, bootstrapNumSlots, bootNumCts,
                        bootNumStc, defaultEncLevel, denseHammingWeight,
                        sparseHammingWeight, logMessageRatio);

    moduleOp->removeAttr(ckks::CKKSDialect::kSchemeParamAttrName);
    moduleOp->removeAttr("scheme.ckks");
  }
};

}  // namespace mlir::heir::cheddar
