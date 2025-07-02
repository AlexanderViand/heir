---
title: Generate BGV Parameters
weight: 1
---

## Overview

The `generate-param-bgv` pass automatically generates BGV
(Brakerski-Gentry-Vaikuntanathan) homomorphic encryption scheme parameters based
on noise analysis of the computation. This pass analyzes the circuit depth and
operations to determine secure and efficient parameters for BGV-based FHE
schemes.

## Input/Output

- **Input**: MLIR module with `secret` dialect operations annotated with
  `mgmt.mgmt` attributes indicating ciphertext levels and management operations
- **Output**: Same module with `bgv.schemeParam` attribute added at module level
  containing generated parameters (logN, Q, P, plaintextModulus)

## Options

- `--model` (string, default: "bgv-noise-kpz21"): Noise model to use for
  parameter generation
  - `bgv-noise-by-bound-coeff-average-case` or `bgv-noise-kpz21`: Uses KPZ21
    coefficient bound model with 2√N expansion factor
  - `bgv-noise-by-bound-coeff-worst-case`: Uses KPZ21 coefficient bound model
    with N expansion factor
  - `bgv-noise-by-variance-coeff` or `bgv-noise-mp24`: Uses MP24 variance
    tracking model for more accurate estimates
  - `bgv-noise-mono`: Uses MMLGA22 canonical embedding model for tighter bounds
- `--plaintext-modulus` (int64, default: 65537): Plaintext modulus for the BGV
  scheme
- `--slot-number` (int, default: 0): Minimum number of slots required for
  parameter generation
- `--use-public-key` (bool, default: true): Whether to use public key encryption
  (affects security parameters)
- `--encryption-technique-extended` (bool, default: false): Whether to use
  extended encryption technique

## Usage Examples

```bash
# Generate BGV parameters with default settings
heir-opt --generate-param-bgv input.mlir

# Use specific noise model with custom plaintext modulus
heir-opt --generate-param-bgv --model=bgv-noise-mp24 --plaintext-modulus=1024 input.mlir

# Generate parameters for specific slot count
heir-opt --generate-param-bgv --slot-number=4096 input.mlir

# Use secret key encryption only
heir-opt --generate-param-bgv --use-public-key=false input.mlir
```

## When to Use

This pass should be used in BGV-based FHE compilation pipelines after:

1. Circuit has been converted to `secret` dialect operations
1. Management operations (`mgmt.relinearize`, `mgmt.modreduce`) have been
   inserted
1. Operations have been annotated with `mgmt.mgmt` attributes (via
   `--annotate-mgmt`)

Typically used before lowering to BGV dialect operations, as the generated
parameters are required for proper BGV operation instantiation.

## Prerequisites

- Input must contain `mgmt` dialect operations and attributes
- Use `--secret-insert-mgmt-bgv` and `--annotate-mgmt` passes before this pass
- Operations must be properly annotated with ciphertext levels and dimensions

## Example Transformation

**Input:**

```mlir
module {
  func.func @add(%arg0: !secret.secret<i16>) -> !secret.secret<i16> {
    %0 = secret.generic(%arg0 : !secret.secret<i16>) attrs = {arg0 = {mgmt.mgmt = #mgmt.mgmt<level = 0>}} {
    ^body(%input0: i16):
      %1 = arith.addi %input0, %input0 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : i16
      secret.yield %1 : i16
    } -> !secret.secret<i16>
    return %0 : !secret.secret<i16>
  }
}
```

**Output:**

```mlir
module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 12, Q = [4294991873], P = [4295049217], plaintextModulus = 65537>} {
  func.func @add(%arg0: !secret.secret<i16>) -> !secret.secret<i16> {
    %0 = secret.generic(%arg0 : !secret.secret<i16>) attrs = {arg0 = {mgmt.mgmt = #mgmt.mgmt<level = 0>}} {
    ^body(%input0: i16):
      %1 = arith.addi %input0, %input0 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : i16
      secret.yield %1 : i16
    } -> !secret.secret<i16>
    return %0 : !secret.secret<i16>
  }
}
```
