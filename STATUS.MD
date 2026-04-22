# HEIR CKKS / Orion Backend Status

Last updated: 2026-04-05

## Scope

This document records the current implementation status of the CKKS / Orion
management and backend-lowering work across:

- Lattigo
- OpenFHE
- CHEDDAR

It also records the current known blocker for OpenFHE auto-scaling modes on the
Orion MLP example, with the relevant source-backed findings.

## Main architectural decisions already landed

### 1. Coarse management is now separated from backend-specific resolution

The CKKS pipeline now has a coarse management phase that:

- annotates structured Orion ops with coarse intrinsic level cost
- inserts ordinary mul-management structure
- inserts explicit merge/reconcile markers at `add` / `sub`
- produces a first candidate `ckks.schemeParam`

This phase does not choose exact target scales or exact add/sub repair.

### 2. Reconcile policy is resolved later

The old early decision point for nominal-vs-CHEDDAR-style cross-level handling
moved out of the coarse insertion pass.

Current policy families:

- local power-of-two compatibility
- canonical-per-level compatibility
- q-aware / BMPH20-style resolution

### 3. Orion impl-style is now an explicit contract

Orion ops carry `orion.impl_style`, and later lowerings must either honor it or
fail loudly.

This is already enforced in:

- `lib/Dialect/LWE/Conversions/LWEToLattigo/LWEToLattigo.cpp`
- `lib/Dialect/LWE/Conversions/LWEToCheddar/LWEToCheddar.cpp`
- `lib/Dialect/LWE/Conversions/LWEToOpenfhe/LWEToOpenfhe.cpp`

Shared verification and impl-style constants now live in:

- `lib/Dialect/Orion/IR/OrionUtils.h`

### 4. OpenFHE CKKS parameter adjustment moved upstream

OpenFHE-specific CKKS chain-length adjustment is no longer supposed to be
rediscovered from lowered OpenFHE IR.

The pass:

- `lib/Dialect/Openfhe/Transforms/AdjustCKKSSchemeParam.cpp`

runs before `ckks-to-lwe` / `lwe-to-openfhe` and:

- reads the current managed CKKS / Orion IR
- models OpenFHE-native CKKS semantics
- enlarges `ckks.schemeParam` if OpenFHE needs a longer chain
- retargets CKKS/LWE types to the final chain
- resolves native Orion linear-transform plaintext levels
- resolves OpenFHE plaintext-encode requirements for `add_plain`

### 4a. Why `AdjustCKKSSchemeParam` currently runs after `secret-to-ckks`

Current placement:

- after the generic CKKS management pipeline has produced managed CKKS / Orion
  IR
- before `ckks-to-lwe`, `lwe-to-openfhe`, and `openfhe-configure-crypto-context`

This is not because the logic belongs in `ConfigureCryptoContext`. It does not.

The current pass sits here because this is the first point in the pipeline where
all of the following are simultaneously true:

- generic CKKS management has already fixed the coarse mgmt structure
- explicit CKKS ops such as `mul`, `rescale`, `level_reduce`, and `add_plain`
  are present in the IR
- opaque Orion ops that will stay opaque for OpenFHE are still visible
- the chosen OpenFHE scaling mode is known
- `ckks.schemeParam` already exists and can be enlarged/retargeted before
  backend lowering

Before `secret-to-ckks`, that information is not all available in one place:

- the IR does not yet contain the actual managed CKKS operations whose OpenFHE
  semantics differ from generic CKKS
- the later bias-encode and native-linear-transform level requirements cannot be
  read off exactly yet

After `lwe-to-openfhe`, it is already too late:

- the scheme parameter has already been committed to lowering
- the CKKS/Orion structure that explains the required OpenFHE state evolution is
  less visible

So the current placement is a principled staging point, not a
`ConfigureCryptoContext` fallback.

However, it is also not the final ideal architecture.

The cleaner long-term solution is:

- generic mgmt computes the backend-agnostic CKKS structure
- an OpenFHE-aware mgmt/populate-style phase models the exact OpenFHE state law
  for the chosen scaling mode
- backend lowering then only reads out that decision

So the current pass should be understood as:

- upstream enough to be in the management part of the pipeline
- but still a compensation layer for functionality that the generic mgmt
  infrastructure does not yet express directly

### 5. ConfigureCryptoContext is now supposed to be dumb for CKKS

For CKKS with `ckks.schemeParam` already present,
`openfhe-configure-crypto-context` should read the scheme parameters and emit
OpenFHE helper functions. It should not silently rediscover CKKS structured-op
depth and take the max.

That cleanup is partially landed already. The remaining issue is not in
`ConfigureCryptoContext`; it is in the upstream OpenFHE CKKS semantics modeled
by `AdjustCKKSSchemeParam`.

## Current backend state

### Lattigo

Current practical state:

- generic CKKS still defaults to nominal management
- Orion `linear_transform` and `chebyshev` lower natively
- BMPH20-style precise handling is the best natural fit for Lattigo, because its
  native polynomial evaluator accepts an explicit target scale

Important source-backed facts:

- ordinary Lattigo CKKS does exact tracked scale management with explicit
  `Rescale`, `SetScale`, and scale matching at add/sub
- native polynomial evaluation accepts an explicit target scale
- native linear-transform support includes stronger reuse contracts for multiple
  transforms on the same ciphertext

### CHEDDAR

Current practical state:

- CHEDDAR uses the canonical-per-level schedule
- Orion `linear_transform` and `chebyshev` lower natively
- CHEDDAR emits/consumes a compiled or canonical-level style for structured ops

Important source-backed facts:

- canonical scale is a function of the chosen level
- native polynomial evaluation is compiled against a fixed target level/scale
- native linear transform carries its own hoisting/BSGS machinery

### OpenFHE

Current practical state:

- `fixed-manual` works on the Orion examples and MLP
- `fixed-auto`, `flexible-auto`, and `flexible-auto-ext` work on smaller Orion
  structured examples
- the remaining failing end-to-end case is the Orion MLP in the auto/flexible
  modes

OpenFHE scaling modes currently in scope:

- `fixed-manual`
- `fixed-auto`
- `flexible-auto`
- `flexible-auto-ext`

The pass / lowering / emitter stack already supports native OpenFHE lowering
for:

- `orion.linear_transform`
- `orion.chebyshev`

There is also explicit Orion lowering to CKKS primitives via:

- `lib/Transforms/LowerOrion/LowerOrion.cpp`

Current Orion impl styles:

- `opaque`
- `diagonal-basic`

## What is currently passing

Local passing coverage already includes:

- `//tests/Dialect/LWE/Conversions/lwe_to_openfhe:all`
- `//tests/Transforms/mlir_to_openfhe_ckks:all`
- `//tests/Examples/orion/linear_transform:all`
- `//tests/Examples/orion/chebyshev:all`
- `//tests/Examples/orion/mlp:mlp_openfhe_test`

The passing OpenFHE MLP target above is the `fixed-manual` one.

The failing OpenFHE MLP targets are:

- `//tests/Examples/orion/mlp:mlp_openfhe_fixed_auto_test`
- `//tests/Examples/orion/mlp:mlp_openfhe_flexible_auto_test`
- `//tests/Examples/orion/mlp:mlp_openfhe_flexible_auto_ext_test`

## Current blocker: OpenFHE auto/flexible MLP

### Symptom

The failing MLP variants throw at runtime with:

`external/openfhe+/src/core/include/lattice/hal/default/poly.h:l.274:operator*=(): Modulus mismatch`

### Exact failing operation

Using a temporary debug build with `insert-debug-handler-calls=true`, the last
successful operation is:

- the second explicit `openfhe.mod_reduce_inplace`

The failure happens at the next operation:

- the third native `openfhe.linear_transform`

### What the debug trace showed

For `fixed-auto`, the runtime state before the failing third native linear
transform is:

- ciphertext scale `2^208`
- ciphertext `Level = 1`
- ciphertext `NoiseScaleDeg = 8`

The emitted native linear-transform plaintext level at that point is:

- `plaintextLevel = 9`

That mismatch is fatal: the native auxiliary plaintexts are encoded against a
different modulus prefix than the ciphertext actually has, and the helper fails
while multiplying them together.

### Why the pass predicted the wrong level

The current upstream OpenFHE CKKS analysis still assumes generic CKKS-style
explicit rescale semantics in places where OpenFHE auto modes do something
different.

The critical source fact is:

- in OpenFHE, generic `ModReduceInPlace` only does work for `FIXEDMANUAL`

Source:

- `external/openfhe+/src/pke/lib/schemerns/rns-leveledshe.cpp`

So for `fixed-auto` / `flexible-auto` / `flexible-auto-ext`:

- lowering `ckks.rescale` to `openfhe.mod_reduce_inplace` does not reduce the
  runtime ciphertext level

The current OpenFHE CKKS adjustment pass still treated those CKKS rescale points
as level-dropping steps in parts of its analysis, so by the time it reached the
third native linear transform it had predicted a lower remaining-Q level than
the runtime actually had.

## What the real OpenFHE law looks like

### 1. `fixed-manual`

This is the only mode where the ordinary explicit `ModReduceInPlace` and
`LevelReduceInPlace` APIs actually implement the expected external level drop.

For this mode:

- explicit CKKS `rescale` / `level_reduce` map naturally to the lowered OpenFHE
  ops

### 2. `fixed-auto`, `flexible-auto`, `flexible-auto-ext`

These modes use OpenFHE’s internal multiply/add alignment logic instead of the
explicit generic `ModReduceInPlace` / `LevelReduceInPlace` path.

Important source-backed behavior:

- `AdjustForMultInPlace` routes non-manual CKKS modes through
  `AdjustLevelsAndDepthToOneInPlace`
- `AdjustLevelsAndDepthToOneInPlace` may do internal reductions before a
  multiply, depending on the current level/depth state
- generic `ModReduceInPlace` itself is a no-op for these modes

Relevant source files:

- `external/openfhe+/src/pke/lib/schemerns/rns-leveledshe.cpp`
- `external/openfhe+/src/pke/lib/scheme/ckksrns/ckksrns-leveledshe.cpp`

### Consequence for HEIR

The OpenFHE-specific CKKS adjustment pass must model the actual OpenFHE
recurrence on the managed CKKS / Orion IR. It cannot reuse the generic CKKS
interpretation of:

- `mul`
- `rescale`
- `level_reduce`

for the auto/flexible modes.

## Concrete source-backed runtime observations from the failing `fixed-auto` MLP

The debug trace produced the following sequence:

- input ciphertext:
  - `Scale = 2^26`
  - `Level = 0`
  - `NoiseScaleDeg = 1`
- after first native linear transform:
  - `Scale = 2^52`
  - `Level = 0`
  - `NoiseScaleDeg = 2`
- after first square and explicit `mod_reduce_inplace`:
  - `Scale = 2^52`
  - `Level = 1`
  - `NoiseScaleDeg = 2`
- after second native linear transform:
  - `Scale = 2^78`
  - `Level = 1`
  - `NoiseScaleDeg = 3`
- after second bias add:
  - `Scale = 2^104`
  - `Level = 1`
  - `NoiseScaleDeg = 4`
- after second square and explicit `mod_reduce_inplace`:
  - `Scale = 2^208`
  - `Level = 1`
  - `NoiseScaleDeg = 8`
- third native linear transform then fails immediately

This confirms:

- the first explicit `mod_reduce_inplace` in `fixed-auto` did not behave like a
  generic CKKS level-drop
- the second explicit `mod_reduce_inplace` also did not behave like a generic
  CKKS level-drop
- the OpenFHE-native state reaching the third linear transform is materially
  different from the one predicted by the current adjustment pass

## Why the initial parameter selection gets it wrong

The coarse CKKS management pipeline computes a modulus chain under HEIR’s
generic CKKS semantics.

That is enough for:

- nominal / power-of-two style management
- CHEDDAR canonical-per-level management
- OpenFHE `fixed-manual`

It is not enough for OpenFHE auto/flexible modes, because those modes do not
realize the generic CKKS `mul + rescale` pattern the same way.

So the first parameter set can be wrong for two separate reasons:

- chain-length demand is different under OpenFHE’s real native state evolution
- native linear-transform auxiliary plaintext levels are different under that
  real state evolution

This is exactly why the OpenFHE-specific adjustment pass exists.

## Current code cleanups already landed

### Shared Orion helpers

Shared in:

- `lib/Dialect/Orion/IR/OrionUtils.h`

This already covers:

- `kOpaqueImplStyle`
- `kDiagonalBasicImplStyle`
- `verifyImplStyle(...)`

### Shared OpenFHE linear-transform helpers

Shared in:

- `lib/Target/OpenFhePke/OpenFheLinearTransformHelpers.h`

This already removed the duplicated linear-transform term grouping logic from:

- `OpenFhePkeEmitter.cpp`
- `Interpreter.cpp`

### Rotation normalization cleanup

`LowerOrion` now uses the shared signed-minimal rotation helper in:

- `lib/Utils/RotationUtils.h`

## Remaining cleanup items

### 1. OpenFHE CKKS analysis state needs to be unified

The current pass still keeps:

- effective level
- noise-scale degree

in separate dataflow analyses.

For OpenFHE auto/flexible multiply semantics, those quantities are coupled.

The next implementation step should be one combined OpenFHE CKKS state lattice
over:

- effective remaining-Q level
- effective noise-scale degree

### 2. OpenFHE auto/flexible recurrence must live in the upstream adjustment pass

The logic belongs in:

- `lib/Dialect/Openfhe/Transforms/AdjustCKKSSchemeParam.cpp`

It does not belong in:

- `ConfigureCryptoContext`

### 3. `ConfigureCryptoContext` should stay a reader/emitter

For CKKS, `ConfigureCryptoContext` should:

- read the finalized `ckks.schemeParam`
- read backend-specific feature requirements already attached to the IR
- emit context/keygen helpers

It should not:

- recompute CKKS structured-op depth
- silently take `max` with a second analysis result
- repair wrong upstream mgmt decisions

### 4. `management.md` still needs a full rewrite pass

The design doc should describe the final intended HEIR management model in one
common language across:

- Lattigo
- OpenFHE fixed/manual/auto/flexible variants
- CHEDDAR

and then describe how HEIR management handles each one.

The current doc has useful content, but it still mixes:

- final architecture
- historical/current implementation details
- transitional explanations

## Immediate next code changes

1. Replace the split OpenFHE level/degree analyses with one combined state.

1. Encode OpenFHE mode-specific multiply/rescale behavior in that analysis.

1. Recompute:

- chain-length requirement
- native linear-transform plaintext level
- `add_plain` encode depth/level requirements

from that combined state.

4. Re-run:

- `//tests/Examples/orion/mlp:mlp_openfhe_fixed_auto_test`
- `//tests/Examples/orion/mlp:mlp_openfhe_flexible_auto_test`
- `//tests/Examples/orion/mlp:mlp_openfhe_flexible_auto_ext_test`

5. Remove the temporary debug-only MLP target once the bug is fixed.

## Verification status

Local verification completed so far:

- OpenFHE CKKS conversion and transform suites pass
- Orion structured OpenFHE examples pass except for the three MLP auto/flexible
  targets above

Still outstanding before this branch is finished:

- fix the OpenFHE auto/flexible MLP bug
- rewrite `management.md` in its final intended form
- run the full local suite expected for the touched areas
- run the full Sky `heir-cheddar` suite
