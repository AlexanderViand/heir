# OpenFHE CKKS State Law

Last updated: 2026-04-05

This note records what the OpenFHE CKKS implementation actually does in the
source tree that HEIR vendors today. It is written as a technical reference for
HEIR management and lowering work.

Source files used heavily:

- `external/openfhe+/src/pke/include/schemerns/rns-cryptoparameters.h`
- `external/openfhe+/src/pke/lib/scheme/ckksrns/ckksrns-cryptoparameters.cpp`
- `external/openfhe+/src/pke/lib/schemerns/rns-leveledshe.cpp`
- `external/openfhe+/src/pke/lib/scheme/ckksrns/ckksrns-leveledshe.cpp`
- `external/openfhe+/src/pke/lib/scheme/ckksrns/ckksrns-fhe.cpp`
- `external/openfhe+/src/pke/lib/scheme/ckksrns/ckksrns-advancedshe.cpp`
- `external/openfhe+/src/pke/lib/keyswitch/keyswitch-hybrid.cpp`
- `external/openfhe+/src/pke/lib/schemebase/base-leveledshe.cpp`

HEIR uses this source law during management and conversion. OpenFHE-specific
metadata (such as `openfhe.native_plaintext_level`) is set in `SecretToCKKS`
when creating `orion.linear_transform` ops, derived from the managed level
assignment in the ciphertext type.

## 1. Core state variables

OpenFHE CKKS ciphertext state is carried in at least these fields:

- `Level`
- `NoiseScaleDeg`
- `ScalingFactor`

The important conventions are:

- `Level` is the number of dropped Q towers.
- Fresh ciphertexts start at `Level = 0`.
- `ModReduceInternalInPlace` increments `Level`.
- `LevelReduceInternalInPlace` increments `Level`.
- `NoiseScaleDeg` is the scale-depth exponent tracked by the runtime.
- Ciphertext-ciphertext multiply sets
  `NoiseScaleDeg(out) = NoiseScaleDeg(lhs) + NoiseScaleDeg(rhs)`.
- Ciphertext-plaintext multiply sets
  `NoiseScaleDeg(out) = NoiseScaleDeg(ct) + NoiseScaleDeg(pt)`.

Source:

- `ckksrns-leveledshe.cpp`, `ModReduceInternalInPlace`
- `ckksrns-leveledshe.cpp`, `LevelReduceInternalInPlace`
- `base-leveledshe.cpp`, `EvalMultCore`
- `rns-leveledshe.cpp`, plaintext multiply paths

### 1.1 Q-count vocabulary

Let:

- `sizeQ` = number of Q moduli in the crypto parameters
- `level` = `ciphertext->GetLevel()`

Then the runtime ciphertext carries:

- `remainingQ = sizeQ - level`

This matters because OpenFHE native linear-transform precompute is parameterized
in terms of remaining-Q count, not in terms of dropped-level count.

## 2. Scaling-factor tables

OpenFHE has two qualitatively different scale-table families.

### 2.1 Constant scaling-factor family

For:

- `FIXEDMANUAL`
- `FIXEDAUTO`

the source returns:

- `GetScalingFactorReal(l) = m_approxSF`
- `GetScalingFactorRealBig(l) = m_approxSF`
- `GetModReduceFactor(l) = m_approxSF`

where `m_approxSF = 2^p`.

Source:

- `rns-cryptoparameters.h`
- `ckksrns-cryptoparameters.cpp`

This is the fixed-delta family.

### 2.2 Level-dependent scaling-factor family

For:

- `FLEXIBLEAUTO`
- `FLEXIBLEAUTOEXT`
- `COMPOSITESCALINGAUTO`
- `COMPOSITESCALINGMANUAL`

the source precomputes per-level tables:

- `m_scalingFactorsReal`
- `m_scalingFactorsRealBig`
- `m_dmoduliQ`

Source:

- `ckksrns-cryptoparameters.cpp`

#### FLEXIBLEAUTO recurrence

For ordinary flexible-auto levels:

- `sf[0] = q_last`
- `sf[k] = sf[k-1]^2 / q[sizeQ - k]`

The source also enforces:

- `0.5 < sf[k] / lastPresetFactor < 2.0`

If the ratio leaves that range, OpenFHE throws and recommends `FIXEDMANUAL` or
`FIXEDAUTO`.

#### FLEXIBLEAUTOEXT initialization

With extra bits:

- `sf[0] = q_last`
- `sf[1] = q[sizeQ - 2]`
- the same recurrence starts from there

Special low-depth cases are handled separately in the source.

#### Composite scaling recurrence

For composite scaling:

- the recurrence runs only on `compositeDegree` boundaries
- non-boundary entries are set to `1`

Source:

- `ckksrns-cryptoparameters.cpp`
- `rns-cryptoparameters.h`

## 2.3 Generic law names used in HEIR

For management purposes, the OpenFHE scaling techniques fall into these generic
law families:

- `FIXEDMANUAL`

- `FIXEDAUTO`

  both use the fixed-delta law

- `FLEXIBLEAUTO`

  uses the q-derived recurrence law

- `FLEXIBLEAUTOEXT`

  uses the q-derived recurrence law with an explicit top pair

- `COMPOSITESCALINGAUTO`

- `COMPOSITESCALINGMANUAL`

  use blockwise q-derived schedules

- `NORESCALE`

  opts out of the rescale law entirely

The important split is:

- the scale-table family
- the runtime state-transition law

`FIXEDMANUAL` and `FIXEDAUTO` are the clearest example: same scale table, very
different runtime level-transition behavior.

For HEIR management, the practical interpretation is:

- `FIXEDMANUAL` is a prescriptive mode
- `FIXEDAUTO`, `FLEXIBLEAUTO`, and `FLEXIBLEAUTOEXT` are predictive modes

In the predictive modes, HEIR does not get to choose how rescale/level
adjustment happens internally. It has to predict what the OpenFHE runtime will
do.

## 3. Scaling mode families

OpenFHE CKKS has three behavior families that matter to HEIR.

### 3.1 `FIXEDMANUAL`

Properties:

- fixed `2^p` scaling factor
- generic `ModReduceInPlace` is real
- generic `LevelReduceInPlace` is real
- external rescale/level-reduce operations are meaningful

Source:

- `rns-leveledshe.cpp`
- `ckksrns-leveledshe.cpp`

### 3.2 `FIXEDAUTO`

Properties:

- same fixed `2^p` scale table as `FIXEDMANUAL`
- generic `ModReduceInPlace` is a no-op
- generic `LevelReduceInPlace` is a no-op
- multiply/add alignment is handled by internal adjustment logic

This is the most important source fact for HEIR:

- `FIXEDAUTO` shares the fixed-delta scale table with `FIXEDMANUAL`
- it does not share the explicit level-transition law

Source:

- `rns-cryptoparameters.h`
- `rns-leveledshe.cpp`

### 3.3 `FLEXIBLEAUTO` and `FLEXIBLEAUTOEXT`

Properties:

- per-level `GetScalingFactorReal(level)`
- per-level `GetModReduceFactor(level) = q[level]`
- generic `ModReduceInPlace` is a no-op
- generic `LevelReduceInPlace` is a no-op
- multiply/add alignment is handled by internal adjustment logic

Source:

- `rns-cryptoparameters.h`
- `ckksrns-cryptoparameters.cpp`
- `rns-leveledshe.cpp`

### 3.4 Composite scaling modes

Properties:

- level-dependent scale tables
- reductions occur in blocks of `compositeDegree`
- generic `ModReduceInPlace` and `LevelReduceInPlace` still do not act like the
  fixed-manual public semantics

Source:

- `rns-cryptoparameters.h`
- `ckksrns-cryptoparameters.cpp`
- `rns-leveledshe.cpp`

### 3.5 `NORESCALE`

Properties:

- add/sub go straight to the core arithmetic
- multiply goes straight to the core arithmetic
- `LevelReduceInPlace` throws

Source:

- `rns-leveledshe.cpp`

## 4. Generic operator transfer rules

This section describes the runtime law that the source implements.

### 4.1 Add/sub, ciphertext-ciphertext

For `NORESCALE`:

- OpenFHE performs direct add/sub.

For all other CKKS scaling modes:

- OpenFHE first calls `AdjustForAddOrSubInPlace`.

Source:

- `rns-leveledshe.cpp`

#### `FIXEDMANUAL`

`AdjustForAddOrSubInPlace` does:

- `AdjustLevelsInPlace(ciphertext1, ciphertext2)`

This is literal tower-count alignment.

When one side is effectively a plaintext-shaped one-component ciphertext, the
code can also scale that side up to the other side's depth using powers of the
fixed scaling factor.

Source:

- `rns-leveledshe.cpp`

#### `FIXEDAUTO`, `FLEXIBLEAUTO`, `FLEXIBLEAUTOEXT`, composite modes

`AdjustForAddOrSubInPlace` does:

- `AdjustLevelsAndDepthInPlace(ciphertext1, ciphertext2)`

This function is the real alignment law for non-manual CKKS add/sub.

The code path depends on:

- `Level(lhs)`
- `Level(rhs)`
- `NoiseScaleDeg(lhs)`
- `NoiseScaleDeg(rhs)`
- `compositeDegree`

The function can:

- scale one operand by a constant
- perform internal mod-reduce
- perform internal level-reduce
- promote a lower-depth operand by multiplying by `1.0`

Source:

- `ckksrns-leveledshe.cpp`

### 4.2 Add/sub, ciphertext-plaintext

OpenFHE first morphs the plaintext into ciphertext form:

- `ctmorphed = MorphPlaintext(plaintext, ciphertext)`

Then it uses the same alignment law as above.

So for non-manual modes:

- plaintext add/sub is also governed by `AdjustLevelsAndDepthInPlace`

Source:

- `rns-leveledshe.cpp`

This is why the encoded plaintext level and encoded plaintext `NoiseScaleDeg`
matter. They change the state of the ciphertext that reaches the add.

#### `FLEXIBLEAUTOEXT` top-of-chain plaintexts

`MakeCKKSPackedPlaintextInternal(..., level = 0, noiseScaleDeg = d)` has a
special path in `FLEXIBLEAUTOEXT`:

- it uses `GetScalingFactorRealBig(0)`
- it ignores the requested `noiseScaleDeg` during encoding
- after encoding it forces the plaintext runtime `NoiseScaleDeg` to `2`

Source:

- `cryptocontext.h`

At level `0`, a plaintext therefore never arrives at `EvalAdd` / `EvalSub` with
runtime degree greater than `2`, no matter what degree the caller asked for.

This interacts directly with `AdjustLevelsAndDepthInPlace`.

If ciphertext and plaintext are both at level `0` and the plaintext has smaller
depth, OpenFHE only performs one:

- `EvalMultCoreInPlace(ciphertext2, 1.0)`

on the lower-depth side.

That raises the lower-depth operand by exactly one degree and multiplies it by
`GetScalingFactorReal(level)`, not by `GetScalingFactorRealBig(level)`.

Source:

- `ckksrns-leveledshe.cpp`

So a level-0 plaintext in `FLEXIBLEAUTOEXT` does not carry an arbitrary caller-
chosen runtime degree. The runtime sees degree `2`, and same-level add/sub can
only promote that lower-depth side once.

This is the key top-pair source fact for HEIR:

- a naive “encode at level 0 with requested degree d” rule is not correct for
  large `d`

If the plaintext meets one level deeper, `AdjustLevelsAndDepthInPlace`:

- multiplies the ciphertext by a scalar
- immediately calls `ModReduceInternalInPlace`
- leaves the ciphertext runtime `NoiseScaleDeg` unchanged overall

So the deeper-plaintext branch is a real candidate mechanism available in the
runtime. But the correct HEIR rule for when to use that branch, and what exact
encode metadata should accompany it, still has to be derived from the full
end-to-end state law rather than from `NoiseScaleDeg` alone.

### 4.3 Multiply, ciphertext-ciphertext

For `NORESCALE`:

- direct multiply core

For `FIXEDMANUAL`:

- `AdjustLevelsInPlace`
- then multiply core
- no automatic level drop

For all other CKKS scaling modes:

- `AdjustForMultInPlace`
- then multiply core

Source:

- `rns-leveledshe.cpp`

#### `AdjustForMultInPlace`

For non-manual modes:

- `AdjustForMultInPlace` calls `AdjustLevelsAndDepthToOneInPlace`

`AdjustLevelsAndDepthToOneInPlace` does:

1. `AdjustLevelsAndDepthInPlace`
1. if the resulting first ciphertext has `NoiseScaleDeg == 2`,
   `ModReduceInternalInPlace` is applied to both operands

Source:

- `rns-leveledshe.cpp`
- `ckksrns-leveledshe.cpp`

This is the key source fact for auto/flexible modes:

- level drops happen inside multiply alignment
- they do not come from the generic external `ModReduceInPlace` op

### 4.4 Multiply, ciphertext-plaintext

OpenFHE:

- morphs the plaintext
- runs `AdjustForMultInPlace(ciphertext, ctmorphed)`
- performs core multiply
- sets `NoiseScaleDeg(out) = NoiseScaleDeg(ct) + NoiseScaleDeg(pt)`

Source:

- `rns-leveledshe.cpp`

For non-manual modes, ciphertext-plaintext multiply therefore also depends on:

- plaintext level
- plaintext `NoiseScaleDeg`
- current ciphertext level/depth

### 4.5 Constant multiply

For CKKS constant multiply by `double` or `complex<double>`:

- in non-manual modes, if `NoiseScaleDeg == 2`, OpenFHE performs
  `ModReduceInternalInPlace` before the multiply
- then `EvalMultCoreInPlace` increments `NoiseScaleDeg` by `1`
- and multiplies the ciphertext scaling factor by
  `GetScalingFactorReal(ciphertext->GetLevel())`

Source:

- `ckksrns-leveledshe.cpp`

### 4.6 Square

For `NORESCALE`, `FIXEDMANUAL`, or `NoiseScaleDeg == 1`:

- square core directly

For other modes when `NoiseScaleDeg == 2`:

- `ModReduceInternalInPlace`
- then square core

Source:

- `rns-leveledshe.cpp`
- `base-leveledshe.cpp`

### 4.7 Generic external `ModReduceInPlace`

The generic external API is mode-sensitive.

The implementation:

- does real work only for `FIXEDMANUAL`

Source:

- `rns-leveledshe.cpp`

For:

- `FIXEDAUTO`
- `FLEXIBLEAUTO`
- `FLEXIBLEAUTOEXT`
- composite scaling modes

the generic external `ModReduceInPlace` does not implement the level drop that
HEIR's ordinary CKKS `rescale` suggests.

### 4.8 Generic external `LevelReduceInPlace`

The generic external API is also mode-sensitive.

The implementation:

- throws for `NORESCALE`
- performs real work only for `FIXEDMANUAL`

Source:

- `rns-leveledshe.cpp`

For auto/flexible/composite modes:

- the generic external `LevelReduceInPlace` is not the runtime state law

## 5. Native linear transform

OpenFHE native linear transform is implemented in `FHECKKSRNS`.

Relevant entry points:

- `EvalLinearTransformPrecompute`
- `EvalLinearTransform`
- `MakeAuxPlaintext`
- `EvalMultExt`
- `EvalAddExtInPlace`
- `KeySwitchExt`
- `KeySwitchDown`
- `KeySwitchDownFirstElement`

Source:

- `ckksrns-fhe.cpp`
- `keyswitch-hybrid.cpp`

### 5.1 Precompute plaintext level

`EvalLinearTransformPrecompute(..., L)` computes:

- `towersToDrop = sizeQ - L - compositeDegree`

Then it pops `towersToDrop` Q towers from the plaintext parameter set before
building the auxiliary plaintexts.

Source:

- `ckksrns-fhe.cpp`

For the common `compositeDegree = 1` case:

- `L = remainingQ - 1`
- equivalently
- `L = sizeQ - Level - 1`

This `L` parameter is the one HEIR needs to predict exactly for native
`linear_transform`.

### 5.2 Aux plaintext scaling factor

`MakeAuxPlaintext(..., level=towersToDrop, noiseScaleDeg)` uses:

- `GetScalingFactorReal(level)`

and then sets the final plaintext scaling factor to:

- `pow(GetScalingFactorReal(level), noiseScaleDeg)`

Source:

- `ckksrns-fhe.cpp`

### 5.3 Runtime transfer law

The runtime evaluation does:

- hoisted rotations on the input
- `EvalMultExt` by each diagonal plaintext
- `EvalAddExtInPlace` to accumulate
- `KeySwitchDown` / `KeySwitchDownFirstElement`

Metadata law:

- `EvalMultExt` adds plaintext `NoiseScaleDeg`
- `EvalMultExt` multiplies the scaling factor
- `EvalAddExtInPlace` does not change metadata
- `KeySwitchExt` / `KeySwitchDown` preserve metadata

Source:

- `ckksrns-fhe.cpp`
- `keyswitch-hybrid.cpp`

For the usual auxiliary plaintexts used here (`noiseScaleDeg = 1`):

- `Level(out) = Level(in)`
- `NoiseScaleDeg(out) = NoiseScaleDeg(in) + 1`
- `ScalingFactor(out) = ScalingFactor(in) * GetScalingFactorReal(Level(in))`

with the `Level(in)` index interpreted through the precompute `L` mapping above.

## 6. Native polynomial evaluation and Chebyshev

OpenFHE native polynomial evaluation is implemented in:

- `ckksrns-advancedshe.cpp`

Relevant entry points:

- `EvalPowers`
- `EvalPoly`
- `EvalPolyPS`
- `EvalChebyPolys`
- `EvalChebyshevSeries`
- `EvalChebyshevSeriesPS`
- `EvalChebyshevSeriesWithPrecomp`

### 6.1 Power generation

The PS path:

- builds powers with `EvalSquare` / `EvalMult`
- calls `ModReduceInPlace`
- sometimes calls `LevelReduceInPlace`
- for non-manual modes, later equalization uses `AdjustLevelsAndDepthInPlace`

Source:

- `ckksrns-advancedshe.cpp`

This means:

- the same high-level PS algorithm has different runtime level/depth behavior
  across scaling modes

### 6.2 Chebyshev preprocessing

The linear Chebyshev-polynomial builder first reduces the domain to `[-1, 1]`
with:

- `alpha = 2 / (b - a)`
- `beta = 2a / (b - a)`
- `T[0] = EvalMult(x, alpha)`
- `ModReduceInPlace(T[0])`
- `EvalAddInPlace(T[0], -1 - beta)`

Source:

- `ckksrns-advancedshe.cpp`

### 6.3 Explicit `ModReduceInPlace` and `LevelReduceInPlace` inside poly eval

The source contains many explicit calls to:

- `cc->ModReduceInPlace(...)`
- `cc->LevelReduceInPlace(...)`

inside the polynomial and Chebyshev evaluators.

In `FIXEDMANUAL` those calls have the obvious literal meaning.

In `FIXEDAUTO`, `FLEXIBLEAUTO`, `FLEXIBLEAUTOEXT`, and composite auto/manual:

- those generic APIs are not the real external level law
- the actual state evolution is dominated by:
  - `EvalMult`
  - `EvalSquare`
  - `EvalAdd`
  - `AdjustLevelsAndDepthInPlace`
  - `AdjustLevelsAndDepthToOneInPlace`

This is visible directly in the source.

## 7. Facts HEIR needs to model

The source above gives the following direct requirements for HEIR.

### 7.1 `FIXEDMANUAL`

HEIR can use a literal public-op model:

- explicit `rescale` maps to generic `ModReduceInPlace`
- explicit `level_reduce` maps to generic `LevelReduceInPlace`
- native structured ops still need correct plaintext levels/degrees

### 7.2 `FIXEDAUTO`

HEIR cannot use the `FIXEDMANUAL` level law.

Facts:

- scale table is still fixed-delta
- generic external `ModReduceInPlace` is a no-op
- generic external `LevelReduceInPlace` is a no-op
- runtime level changes happen in multiply alignment and native op internals

### 7.3 `FLEXIBLEAUTO` / `FLEXIBLEAUTOEXT`

HEIR must model both:

- the per-level scaling-factor table
- the auto alignment law

Facts:

- `GetScalingFactorReal(level)` is not constant
- `GetModReduceFactor(level)` is the dropped modulus
- generic external `ModReduceInPlace` is a no-op
- generic external `LevelReduceInPlace` is a no-op

### 7.4 Native `linear_transform`

HEIR must predict:

- the exact precompute `L`
- the exact runtime output `NoiseScaleDeg`
- the exact runtime output scale law

The required precompute formula in the common non-composite case is:

- `L = sizeQ - Level(in) - 1`

### 7.5 Native `chebyshev` / `poly_eval`

HEIR cannot treat native polynomial evaluation as a black box with a single
"cost" number if the goal is exact OpenFHE parity.

The source shows:

- internal multiplies, squares, level reductions, mod reductions, and add
  equalization
- mode-sensitive behavior at all of those points

## 8. Practical consequence for HEIR architecture

If HEIR wants OpenFHE-native behavior to be predicted exactly, then the
OpenFHE-aware management logic must model:

- scaling-mode family
- runtime `Level`
- runtime `NoiseScaleDeg`
- native structured-op contracts

Doing only generic CKKS mgmt and then assuming the public `rescale` /
`level_reduce` law survives unchanged into OpenFHE auto/flexible modes is not
correct.

## 9. Why generic CKKS parameter selection can be too small

Generic CKKS parameter selection budgets from:

- ct-ct multiply depth
- explicit rescale / level-reduce structure
- coarse opaque-op level cost

That misses OpenFHE-native requirements that are not expressed as ordinary CKKS
level drops.

Examples:

- native `linear_transform` has coarse level cost `0`, but it increases
  `NoiseScaleDeg` by one through auxiliary plaintext multiplication
- non-manual modes move levels inside multiply/add alignment rather than through
  the public `ModReduceInPlace` / `LevelReduceInPlace` APIs
- native `linear_transform` precompute in the non-EXT modes needs one extra Q
  modulus above the ciphertext level used by the transform

So a modulus chain can be large enough for generic CKKS and still be too short
for OpenFHE native execution.

The management consequence is:

- generic CKKS management chooses the coarse structure
- OpenFHE-aware CKKS management then checks the chosen scaling mode against the
  actual OpenFHE runtime law
- if needed, it grows `ckks.schemeParam` before backend lowering
