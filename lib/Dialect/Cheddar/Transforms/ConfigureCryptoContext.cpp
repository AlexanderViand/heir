#include "lib/Dialect/Cheddar/Transforms/ConfigureCryptoContext.h"

#include <algorithm>
#include <string>

#include "lib/Analysis/RotationAnalysis/RotationAnalysis.h"
#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/Cheddar/IR/CheddarOps.h"
#include "lib/Dialect/Cheddar/IR/CheddarTypes.h"
#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Utils/TransformUtils.h"
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Bufferization/IR/Bufferization.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StaticValueUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"           // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"         // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"       // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"      // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"     // from @llvm-project

namespace mlir::heir::cheddar {

#define GEN_PASS_DEF_CHEDDARCONFIGURECRYPTOCONTEXT
#include "lib/Dialect/Cheddar/Transforms/ConfigureCryptoContext.h.inc"

namespace {

// Build a `<entry>__configure() -> (context, user_interface)` function in
// destination-passing tensor form:
//   %p   = cheddar.make_parameter ...
//   %ctx = cheddar.create_context %p, %ctx_init
//   %ui0 = cheddar.create_user_interface %ctx, %ui_init
//   %ui1 = cheddar.prepare_rot_key %ui0 {distance, maxLevel}   // per distance
//   return %ctx, %uiN
// Params come from the CKKS scheme attrs (logN/scale/Q/P); the rotation
// distances come from the shared RotationAnalysis (the same way the lattigo and
// openfhe backends discover which keys to generate). The two tensor results
// become owning context/user_interface out-params after one-shot-bufferize +
// buffer-results-to-out-params, and the cheddar-to-emitc boundary re-types them
// to owning smart-pointer references.
// When the program bootstraps, the generated configure builds a BootContext
// (and runs the one-time bootstrap precompute + rotation-key request) instead
// of a plain Context: the entry then takes a `!cheddar.boot_context`.
// `numSlots` is the slot count the (full-slot) bootstrap refreshes;
// `numCtsLevels` / `numStcLevels` are the CtS/StC level budgets (a
// depth/rotations trade-off, cf. OpenFHE's level-budget-encode/decode),
// threaded in as pass options.
void buildConfigureFunc(ModuleOp moduleOp, func::FuncOp entry, int64_t logN,
                        int64_t logScale, DenseI64ArrayAttr Q,
                        DenseI64ArrayAttr P, ArrayRef<int64_t> rotationIndices,
                        ArrayRef<std::pair<int64_t, int64_t>> ltRotationKeys,
                        bool bootstraps, int64_t numSlots, int64_t numCtsLevels,
                        int64_t numStcLevels, int64_t defaultEncLevel,
                        int64_t denseHammingWeight, int64_t sparseHammingWeight,
                        int64_t logMessageRatio) {
  MLIRContext *ctx = moduleOp.getContext();
  int64_t maxLevel = Q.size() - 1;
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
  // these models bootstrap
  // (|m| ~ O(1), the sign/ReLU inputs), CHEDDAR's default headroom of 5 is the
  // right magnitude (log_scaleup_ ~= 10). Override via the `log-message-ratio`
  // option for a model with a larger message bound.
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
  auto funcType = FunctionType::get(ctx, {}, {ctxTensor, uiTensor});
  // Discovered by name convention (`<entry>__configure`), like the lattigo
  // backend -- no marker attribute needed.
  std::string name = (entry.getSymName() + "__configure").str();
  auto configFunc = func::FuncOp::create(builder, loc, name, funcType);
  configFunc.setPublic();

  Block *bodyBlock = configFunc.addEntryBlock();
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
  // DPS destination buffers; bufferization hoists these allocs to the
  // out-params.
  Value ctxInit = bufferization::AllocTensorOp::create(builder, loc, ctxTensor,
                                                       ValueRange{})
                      .getResult();
  Value context =
      bootstraps
          ? CreateBootContextOp::create(
                builder, loc, TypeRange{ctxTensor}, params, i64(numCtsLevels),
                i64(numStcLevels), i64(effLogMessageRatio), ctxInit)
                ->getResult(0)
          : CreateContextOp::create(builder, loc, TypeRange{ctxTensor},
                                    ValueRange{params, ctxInit})
                ->getResult(0);
  Value uiInit =
      bufferization::AllocTensorOp::create(builder, loc, uiTensor, ValueRange{})
          .getResult();
  Value ui = CreateUserInterfaceOp::create(builder, loc, TypeRange{uiTensor},
                                           ValueRange{context, uiInit})
                 ->getResult(0);
  for (int64_t d : rotationIndices)
    ui = PrepareRotKeyOp::create(builder, loc, TypeRange{uiTensor}, ui, i64(d),
                                 i64(maxLevel), /*chainMaxLevel=*/IntegerAttr())
             ->getResult(0);
  // Bootstrap precompute has the largest transient GPU-memory footprint. Run
  // it before preparing the (potentially hundreds of) linear-transform keys;
  // otherwise those resident keys can make an otherwise-valid N=16 context
  // OOM during CtS/StC precomputation. Both CHEDDAR forks safely add or widen
  // the transform keys afterward, and all keys still land in the same EvkMap.
  if (bootstraps)
    ui = PrepareBootstrapOp::create(builder, loc, TypeRange{uiTensor}, context,
                                    ui, i64(numSlots))
             ->getResult(0);
  // cheddar.linear_transform evaluates its BSGS rotations at the op's level;
  // CHEDDAR's level-specific key lookup (best-fit on the key-switch config)
  // can reject a chain-max key for a much lower level, so prepare each
  // transform's rotations at its actual usage level. The caller has already
  // removed pairs covered by the chain-max loop above. chainMaxLevel makes
  // the emitter dispatch per fork (scale-snu prepares at chain max instead).
  for (auto [d, level] : ltRotationKeys) {
    ui = PrepareRotKeyOp::create(builder, loc, TypeRange{uiTensor}, ui, i64(d),
                                 i64(level), i64(maxLevel))
             ->getResult(0);
  }
  func::ReturnOp::create(builder, loc, ValueRange{context, ui});
}

}  // namespace

struct CheddarConfigureCryptoContext
    : public impl::CheddarConfigureCryptoContextBase<
          CheddarConfigureCryptoContext> {
  using CheddarConfigureCryptoContextBase::CheddarConfigureCryptoContextBase;

  void runOnOperation() override {
    auto moduleOp = cast<ModuleOp>(getOperation());
    MLIRContext *ctx = &getContext();

    // RotationAnalysis requires -sccp to have propagated constants so the
    // rotation indices are statically detectable (mirrors the lattigo/openfhe
    // configure passes).
    OpPassManager pipeline("builtin.module");
    pipeline.addPass(createSCCPPass());
    pipeline.addPass(createCanonicalizerPass());
    (void)runPipeline(pipeline, moduleOp);

    auto schemeParamAttr = moduleOp->getAttrOfType<ckks::SchemeParamAttr>(
        ckks::CKKSDialect::kSchemeParamAttrName);

    if (schemeParamAttr) {
      int64_t logN = schemeParamAttr.getLogN();
      int64_t logDefaultScale = schemeParamAttr.getLogDefaultScale();
      // CKKS SchemeParam already types Q/P as DenseI64ArrayAttr.
      DenseI64ArrayAttr Q = schemeParamAttr.getQ();
      DenseI64ArrayAttr P = schemeParamAttr.getP();

      moduleOp->setAttr("cheddar.logN",
                        IntegerAttr::get(IntegerType::get(ctx, 64), logN));
      moduleOp->setAttr(
          "cheddar.logDefaultScale",
          IntegerAttr::get(IntegerType::get(ctx, 64), logDefaultScale));

      // CHEDDAR derives its RNS limbs from the CKKS Q/P moduli directly;
      // shallow computations must already carry enough primes via scheme
      // management.
      if (Q) moduleOp->setAttr("cheddar.Q", Q);
      if (P) moduleOp->setAttr("cheddar.P", P);

      // Generate the <entry>__configure function so the backend harness can set
      // up the context/keys with one call.
      auto entry = detectEntryFunction(moduleOp, entryFunction);
      if (entry && Q && P) {
        // Discover the rotation keys to generate, the same way the other
        // backends do (RotationAnalysis over the program's rotation ops).
        RotationAnalysis rotationAnalysis;
        if (failed(rotationAnalysis.run(moduleOp))) {
          entry->emitOpError("failed to compute static rotation indices");
          signalPassFailure();
          return;
        }
        const auto &indexSet = rotationAnalysis.getRotationIndices();
        SmallVector<int64_t> rotationIndices(indexSet.begin(), indexSet.end());
        llvm::sort(rotationIndices);  // deterministic key-prep order

        // (rotation, level) pairs for linear_transform ops: their rotations
        // run at the op's level and need level-specific keys under CHEDDAR's
        // best-fit key lookup.
        // Inline (non-prepared) linear_transform ops: their rotations run at
        // the op's level and need level-specific keys. They have no runtime
        // handle, so __configure always emits their keys.
        SetVector<std::pair<int64_t, int64_t>> ltKeySet;
        moduleOp.walk([&](LinearTransformOp ltOp) {
          int64_t ltLevel = ltOp.getLevel().getInt();
          for (OpFoldResult idx : ltOp.getRotationIndices()) {
            if (auto attr = dyn_cast<Attribute>(idx))
              ltKeySet.insert({cast<IntegerAttr>(attr).getInt(), ltLevel});
          }
        });
        // Split-preprocessed (prepared) transforms hand the harness owning
        // shared_ptr<LinearTransform> handles. ALWAYS record their (rotation,
        // level) pairs: the ltOnly() classifier below uses this set to keep
        // these rotations OUT of the chain-max maxKeyRotations set. (Dropping
        // them from the classification -- as a naive `if (!defer) walk` does --
        // silently reclassifies them as generic rotations and emits
        // full-modulus keys for each, which is the transform-heavy GPU keygen
        // OOM.) Whether
        // __configure *emits* per-level keys for them is the deferLintransKeys
        // choice: when deferring, the harness requests exactly the pruned
        // rotations each prepared transform needs at runtime
        // (AddRequiredRotations), at the transform's own level -- so we record
        // but do not emit here.
        SetVector<std::pair<int64_t, int64_t>> preparedLtKeySet;
        moduleOp.walk([&](ApplyPreparedLinearTransformOp ltOp) {
          int64_t ltLevel = ltOp.getLevel().getInt();
          for (OpFoldResult idx : ltOp.getRotationIndices()) {
            if (auto attr = dyn_cast<Attribute>(idx))
              preparedLtKeySet.insert(
                  {cast<IntegerAttr>(attr).getInt(), ltLevel});
          }
        });
        // Keys emitted by __configure: inline always; prepared only when NOT
        // deferring (deferral leaves prepared keys to the harness at runtime).
        SetVector<std::pair<int64_t, int64_t>> emittedLtKeySet(ltKeySet);
        if (!deferLintransKeys)
          for (auto p : preparedLtKeySet) emittedLtKeySet.insert(p);
        SmallVector<std::pair<int64_t, int64_t>> ltRotationKeys(
            emittedLtKeySet.begin(), emittedLtKeySet.end());
        llvm::sort(ltRotationKeys);

        // Rotations used by any non-linear-transform op (hrot & co. look
        // their keys up without level constraints) keep a chain-max key.
        // Rotations used ONLY by linear transforms are keyed at each
        // transform's own level instead: duplicating them at chain max
        // dominates key material on deep bootstrapping circuits (criteo:
        // ~36 GiB of keys, GPU OOM). If any non-LT rotation index cannot be
        // resolved statically (or is negative, i.e. not in the analysis'
        // normalized form), fall back to chain-max keys for everything.
        DenseSet<int64_t> nonLtRotations;
        bool keepAllMaxKeys = false;
        moduleOp.walk([&](RotationOpInterface rotOp) {
          if (isa<LinearTransformOp, ApplyPreparedLinearTransformOp>(
                  rotOp.getOperation()))
            return;
          for (OpFoldResult idx : rotOp.getRotationIndices()) {
            std::optional<int64_t> d;
            if (auto attr = dyn_cast<Attribute>(idx))
              d = cast<IntegerAttr>(attr).getInt();
            else
              d = getConstantIntValue(cast<Value>(idx));
            if (!d.has_value() || *d < 0) {
              keepAllMaxKeys = true;
              return;
            }
            nonLtRotations.insert(*d);
          }
        });
        // Classify using ALL linear-transform rotations (inline + prepared,
        // regardless of deferral) so a prepared-transform rotation is never
        // mistaken for a generic rotation and keyed at chain max.
        DenseSet<int64_t> ltRotationSet;
        for (auto [d, level] : ltKeySet) ltRotationSet.insert(d);
        for (auto [d, level] : preparedLtKeySet) ltRotationSet.insert(d);
        auto ltOnly = [&](int64_t d) {
          return !keepAllMaxKeys && ltRotationSet.contains(d) &&
                 !nonLtRotations.contains(d);
        };

        // Bootstrapping programs need a BootContext + the one-time boot
        // precompute. Detect by the presence of cheddar.boot (lwe-to-cheddar
        // has already run at this point).
        bool bootstraps = false;
        moduleOp.walk([&](BootOp) { bootstraps = true; });
        // Full-slot bootstrap refreshes the context's slot count; the lowering
        // records it as the `scheme.actual_slot_count` module attribute.
        int64_t numSlots = 0;
        if (bootstraps) {
          if (auto slotsAttr = moduleOp->getAttrOfType<IntegerAttr>(
                  kActualSlotCountAttrName))
            numSlots = slotsAttr.getInt();
          else {
            entry->emitOpError(
                "bootstrapping program is missing the scheme.actual_slot_count "
                "module attribute needed to configure the boot context");
            signalPassFailure();
            return;
          }
        }
        // Bootstrap parameters: GenerateParamCKKS sized the modulus chain for
        // the boot circuit and recorded the split (num_cts/num_stc) +
        // default_encryption_level it assumed. Prefer those over the
        // pass-option defaults so the chain, the BootParameter, and the
        // Parameter agree (cheddar's BootContext asserts default_enc == max -
        // num_cts - evalMod).
        int64_t bootNumCts = numCtsLevels;
        int64_t bootNumStc = numStcLevels;
        int64_t defaultEncLevel = static_cast<int64_t>(Q.size()) - 1;
        int64_t denseHammingWeight = 0;
        int64_t sparseHammingWeight = 0;
        if (bootstraps) {
          if (auto a =
                  moduleOp->getAttrOfType<IntegerAttr>("cheddar.boot.num_cts"))
            bootNumCts = a.getInt();
          if (auto a =
                  moduleOp->getAttrOfType<IntegerAttr>("cheddar.boot.num_stc"))
            bootNumStc = a.getInt();
          if (auto a = moduleOp->getAttrOfType<IntegerAttr>(
                  "cheddar.boot.default_encryption_level"))
            defaultEncLevel = a.getInt();
          // CHEDDAR's reference bootstrap params use a dense (full) secret-key
          // hamming weight of N/2 and a sparse hamming weight of 32.
          denseHammingWeight = int64_t{1} << (logN - 1);
          sparseHammingWeight = 32;
        }
        // Chain-max keys only for rotations some non-LT op uses; LT-only
        // rotations are keyed per (rotation, LT level) below (including
        // chain-max LTs).
        int64_t maxLevel = static_cast<int64_t>(Q.size()) - 1;
        SmallVector<int64_t> maxKeyRotations;
        for (int64_t d : rotationIndices)
          if (!ltOnly(d)) maxKeyRotations.push_back(d);
        SmallVector<std::pair<int64_t, int64_t>> ltLevelKeys;
        for (auto [d, level] : ltRotationKeys) {
          if (level == maxLevel && !ltOnly(d)) continue;  // covered above
          ltLevelKeys.push_back({d, level});
        }
        buildConfigureFunc(moduleOp, entry, logN, logDefaultScale, Q, P,
                           maxKeyRotations, ltLevelKeys, bootstraps, numSlots,
                           bootNumCts, bootNumStc, defaultEncLevel,
                           denseHammingWeight, sparseHammingWeight,
                           logMessageRatio);
      }

      // Remove the CKKS scheme param attribute — consumed
      moduleOp->removeAttr(ckks::CKKSDialect::kSchemeParamAttrName);
    }

    // Remove scheme.ckks marker attribute
    moduleOp->removeAttr("scheme.ckks");
  }
};

}  // namespace mlir::heir::cheddar
