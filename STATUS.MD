# HEIR CKKS / Orion Status

Last updated: 2026-04-05

## Current architecture

### Generic CKKS management

The generic CKKS management pipeline now owns the scheme-agnostic work:

- annotate structured Orion ops with coarse intrinsic level cost
- insert mul-side management structure
- insert `mgmt.reconcile` markers at secret `add` / `sub`
- generate an initial `ckks.schemeParam`
- resolve exact scale / reconcile decisions for the selected management law

This runs before `secret-to-ckks`.

### OpenFHE predictive resolution

OpenFHE needs one more upstream stage on managed CKKS / Orion IR:

- [`lib/Transforms/PopulateScale/ResolveOpenfheCKKSMetadata.cpp`](/Users/a.viand/Code/heir/lib/Transforms/PopulateScale/ResolveOpenfheCKKSMetadata.cpp)

That pass is part of management. It runs before `ckks-to-lwe` and before
`lwe-to-openfhe`. It does three jobs:

1. Read the current `ckks.schemeParam`, materialize the actual OpenFHE
   parameters for the selected scaling technique, and update the HEIR module to
   match the chain OpenFHE will actually use.
1. Predict the OpenFHE-specific metadata laws that the generic CKKS pipeline
   does not model directly, including native linear-transform plaintext levels
   and plaintext encoding metadata for `add_plain`.
1. Rewrite CKKS / LWE types and module attrs so later passes see the final
   OpenFHE-compatible chain rather than the provisional generic one.

This pass is not a post-lowering repair. It is the OpenFHE-specific part of
management.

### `ConfigureCryptoContext`

For CKKS,
[`lib/Dialect/Openfhe/Transforms/ConfigureCryptoContext.cpp`](/Users/a.viand/Code/heir/lib/Dialect/Openfhe/Transforms/ConfigureCryptoContext.cpp)
is now just a readout/configuration stage.

It should:

- read the finalized `ckks.schemeParam`
- read the selected OpenFHE scaling technique
- read which features require evaluation keys
- emit OpenFHE parameter/context/keygen helpers

It should not:

- recompute CKKS chain length
- redo CKKS / Orion management analysis
- silently paper over upstream management mistakes

The CKKS path now mostly follows that rule. Any remaining CKKS-specific logic
there should keep moving out, not grow.

### OpenFHE linear transform helper

[`lib/Target/OpenFhePke/OpenFheLinearTransformHelpers.h`](/Users/a.viand/Code/heir/lib/Target/OpenFhePke/OpenFheLinearTransformHelpers.h)
is runtime code, not a second management pass.

It exists because generated code targets the public `CryptoContext` API, while
OpenFHE’s higher-level linear-transform entry points are not exposed through
that surface. The helper rebuilds the OpenFHE linear-transform kernel from the
public pieces that are exposed:

- `EvalFastRotationPrecompute`
- `EvalFastRotationExt`
- `KeySwitchExt`
- `KeySwitchDown`

The helper consumes the already-resolved `openfhe.native_plaintext_level`. It
does not choose levels, scales, or parameters.

### What changed from the old design

The old downstream `AdjustCKKSSchemeParam` pass no longer belongs in the
pipeline. Its logic has been moved upstream into
`ResolveOpenfheCKKSMetadata.cpp`, where it can still see managed CKKS / Orion IR
and make the final OpenFHE prediction before backend lowering.

That is the right split:

- generic mgmt computes the scheme-agnostic skeleton
- OpenFHE-specific mgmt predicts OpenFHE’s actual chain and metadata
- backend lowering consumes those decisions

## Backend status

### Lattigo

Current state:

- ordinary CKKS uses explicit tracked scales
- Orion `linear_transform` lowering works
- Orion `chebyshev` lowering works
- the precise/BMPH20-style path is the natural fit for opaque polynomial
  evaluation because Lattigo lets HEIR choose target output scale directly

### CHEDDAR

Current state:

- ordinary CKKS follows the backend canonical scale schedule
- Orion `linear_transform` lowering works
- Orion `chebyshev` lowering works
- CHEDDAR configuration reads the chain chosen by management

### OpenFHE

Current state:

- Orion `linear_transform` lowering works
- Orion `chebyshev` lowering works
- lowering Orion to ordinary CKKS also exists through
  [`lib/Transforms/LowerOrion/LowerOrion.cpp`](/Users/a.viand/Code/heir/lib/Transforms/LowerOrion/LowerOrion.cpp)
- OpenFHE-specific chain / metadata prediction now happens upstream in
  `ResolveOpenfheCKKSMetadata.cpp`

Scaling techniques currently in active use:

- `fixed-manual`
- `flexible-auto`
- `flexible-auto-ext`

`fixed-auto` is still present in some code paths, but it is no longer the target
design. The meaningful split is:

- prescriptive: `fixed-manual`
- predictive: `flexible-auto`, `flexible-auto-ext`

## Local test status

Passing locally:

- `//tests/Dialect/LWE/Conversions/lwe_to_openfhe:all`
- `//tests/Transforms/mlir_to_openfhe_ckks:all`
- `//tests/Examples/orion/linear_transform:all`
- `//tests/Examples/orion/chebyshev:all`
- `//tests/Examples/orion/mlp:mlp_openfhe_test`
- `//tests/Examples/orion/mlp:mlp_openfhe_flexible_auto_test`

Current failing end-to-end target:

- `//tests/Examples/orion/mlp:mlp_openfhe_flexible_auto_ext_test`

## Current OpenFHE blocker

### Failing case

The only remaining OpenFHE end-to-end failure is the Orion MLP under
`flexible-auto-ext`.

The failure is no longer:

- wrong chain length
- wrong native linear-transform plaintext level
- tower mismatch in native linear transform

What remains is one wrong top-of-chain plaintext-add prediction.

### Observed runtime behavior

The current debug trace shows:

- first native linear transform: correct
- first bias add: correct
- second native linear transform: correct
- second bias add: correct
- third native linear transform: correct
- final bias add: wrong

Observed state at the failing point:

- after the third native linear transform:
  - value about `0.125`
  - `Level = 0`
  - `NoiseScaleDeg = 15`
  - `Scale ~= 2^319`
- after the final `add_plain` with bias `0.25`:
  - the result stays at the transform output instead of adding the bias

So the current upstream predictor still gets the final `flexible-auto-ext`
plaintext encoding wrong.

### Source-backed law behind the bug

From OpenFHE source:

- `MakeCKKSPackedPlaintext(..., level = 0)` in `FLEXIBLEAUTOEXT` does not behave
  like a general-level encode.
- It uses `GetScalingFactorRealBig(0)`.
- It ignores the requested `noiseScaleDeg` during encoding.
- It always produces runtime plaintext `NoiseScaleDeg = 2`.

From the add/sub alignment code:

- same-level depth equalization can only promote the lower-depth side by one
  step with `EvalMultCoreInPlace(..., 1.0)`
- that promotion uses `GetScalingFactorReal(level)`, not
  `GetScalingFactorRealBig(level)`

The combination matters at the top pair:

- a level-0 `flexible-auto-ext` plaintext encode is special
- copying the ciphertext degree into the plaintext request is not enough
- the final MLP bias add exposes exactly that missing law

Relevant sources:

- `external/openfhe+/src/pke/include/cryptocontext.h`
- `external/openfhe+/src/pke/lib/schemerns/rns-leveledshe.cpp`
- `external/openfhe+/src/pke/lib/scheme/ckksrns/ckksrns-leveledshe.cpp`

## Important OpenFHE findings already established

### Public mod-reduce / level-reduce are not the predictive law

For OpenFHE CKKS:

- `ModReduceInPlace` does real public work only in `fixed-manual`
- `LevelReduceInPlace` does real public work only in `fixed-manual`

In predictive modes, real level/depth changes happen inside:

- `AdjustLevelsAndDepthInPlace`
- `AdjustLevelsAndDepthToOneInPlace`
- native structured-op internals

### Native linear-transform plaintext level

The native linear-transform precompute contract is:

- `towersToDrop = sizeQ - L - compositeDegree`

For the common `compositeDegree = 1` case:

- `L = remainingQ - 1`
- equivalently `L = sizeQ - Level - 1`

That is the quantity HEIR resolves into `openfhe.native_plaintext_level`.

### Generic CKKS budgeting is not sufficient by itself

Generic CKKS management budgets from:

- ct-ct multiply depth
- explicit rescale / level-reduce structure
- structured-op coarse level cost

OpenFHE also depends on metadata laws that are specific to the selected scaling
technique and to native structured ops. Those laws must be predicted upstream on
managed CKKS / Orion IR before backend lowering.

## Immediate next steps

1. Finish the upstream `flexible-auto-ext` plaintext-add predictor.
   - Extend `ResolveOpenfheCKKSMetadata.cpp` with the missing top-pair state.
   - Stop treating `NoiseScaleDeg + effective HEIR level` as sufficient for the
     final EXT encode choice.
1. Keep all CKKS/OpenFHE management logic upstream.
   - No new post-`secret-to-ckks` repair pass.
   - No new CKKS chain/depth fallback in `ConfigureCryptoContext`.
1. Rerun the focused OpenFHE suites after the EXT fix:
   - `//tests/Examples/orion/mlp:mlp_openfhe_test`
   - `//tests/Examples/orion/mlp:mlp_openfhe_flexible_auto_test`
   - `//tests/Examples/orion/mlp:mlp_openfhe_flexible_auto_ext_test`
   - `//tests/Dialect/LWE/Conversions/lwe_to_openfhe:all`
   - `//tests/Transforms/mlir_to_openfhe_ckks:all`
1. Tighten the docs:
   - `management.md` should describe the intended final management design
     directly
   - `OPENFHE.MD` should remain the source-backed OpenFHE law reference
1. After the local OpenFHE work is stable:
   - run broader local CKKS/OpenFHE coverage
   - run the full Sky suite on `heir-cheddar`
   - get one final independent review on the completed tree

## Known cleanup still worth doing

- move any remaining duplicate Orion impl-style helpers to shared utilities
- move any remaining duplicate linear-transform grouping helpers to a shared
  header
- review whether any low-level OpenFHE tests now only check copied metadata and
  should be replaced or supplemented with higher-level mgmt tests
