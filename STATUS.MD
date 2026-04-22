# HEIR CKKS / Orion Status

Last updated: 2026-04-06

## Current architecture

### Structured ops at arith level

Structured operations (linear transform, Chebyshev polynomial evaluation) can
now be represented at the arith level and preserved through the management
pipeline:

- `polynomial.eval` with Chebyshev coefficients + domain attributes
- `tensor_ext.diagonal_matvec` with diagonal matrix data + indices

Both are annotated with `orion.level_cost_ub` by `AnnotateOrion`, which lets the
management pipeline budget their level consumption without expanding them.

The `--preserve-structured-ops` flag on `--mlir-to-ckks` /
`--torch-linalg-to-ckks` controls whether these ops are preserved or expanded to
basic arithmetic.

### The raise pass (`--raise-ckks`)

Converts Orion CKKS-level IR back to arith-level secret IR:

- `orion.chebyshev` → `polynomial.eval`
- `orion.linear_transform` → `linalg.matvec` (constant diags) or
  `tensor_ext.diagonal_matvec` (function-arg diags)
- `ckks.{add,sub,mul}` → `arith.{addf,subf,mulf}`
- `ckks.rotate` → `tensor_ext.rotate`
- `ckks.{add_plain,sub_plain,mul_plain}` → arith ops (folds `lwe.rlwe_encode`)
- `ckks.{rescale,relinearize}` → stripped
- Strips `ckks.schemeParam` (re-derived by `generate-param-ckks`)

The raised IR uses plain tensor types with `{secret.secret}` arg attributes,
compatible with `--mlir-to-ckks` (which calls `wrap-generic` to create
`secret.generic`).

### Management pipeline

The generic CKKS management pipeline owns the scheme-agnostic work:

- annotate structured ops with coarse intrinsic level cost
- insert mul-side management structure (`mgmt.modreduce`, `mgmt.relinearize`)
- insert `mgmt.reconcile` markers at secret `add` / `sub`
- generate an initial `ckks.schemeParam`
- resolve exact scale / reconcile decisions

This runs before `secret-to-ckks`. No backend-specific metadata resolution is
needed in the management pipeline. Backend-specific attributes
(`openfhe.native_plaintext_level`) are set during `secret-to-ckks` when the
structured ops are converted to their CKKS-level equivalents.

### Secret-to-CKKS conversion

Converts managed arith-level IR to CKKS scheme IR:

- `polynomial.eval` (Chebyshev) → `orion.chebyshev`
- `tensor_ext.diagonal_matvec` → `orion.linear_transform`
- `arith.{addf,mulf,...}` → `ckks.{add,mul,...}`

The `openfhe.native_plaintext_level` attribute is set here (derived from the
ciphertext type's modulus chain level), not in a separate metadata pass.

## Backend status

### Lattigo

- Ordinary CKKS with explicit tracked scales
- Orion `linear_transform` and `chebyshev` via native API
- Precise/BMPH20-style scale tracking supported
- Raise path tested: raise → re-manage → Lattigo native APIs

### CHEDDAR

- CKKS with `default-scale-schedule` reconcile policy
- Orion `linear_transform` and `chebyshev` via native API
- Uses `ckks-scale-policy=precise` for several test programs

### OpenFHE

- Orion `linear_transform` and `chebyshev` via native API
- Scaling techniques tested: `fixed-manual`, `flexible-auto`,
  `flexible-auto-ext`
- Raise path tested: raise → re-manage → OpenFHE native APIs
- Known limitation: multi-layer MLP (2+ muls with linear_transforms) triggers
  `noiseScaleDeg` accumulation in the EXT-based linear_transform helper, causing
  `SetLevel` failure at runtime. Single-layer circuits work. See OPENFHE.MD
  section 12 for details and possible fixes.

## Compilation paths

### Basic ops path (default)

All structured ops expanded to basic CKKS primitives (rotate + multiply + add).
Works with any backend.

### Native API path (`--preserve-structured-ops`)

Structured ops preserved through management and converted to native backend ops
at `secret-to-ckks` time. Uses backend-specific APIs for better performance.

### Raise + retarget (`--raise-ckks`)

Orion CKKS-level IR raised to arith level, then re-managed through the standard
pipeline for any target backend. Enables using Orion-generated networks with
backends that Orion doesn't directly support.
