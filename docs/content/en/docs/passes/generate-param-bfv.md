---
title: Generate BFV Parameters
weight: 2
---

## Overview

The `generate-param-bfv` pass automatically generates BFV
(Brakerski-Fan-Vercauteren) homomorphic encryption scheme parameters based on
noise analysis of the computation. This pass analyzes the circuit operations to
determine secure and efficient parameters for BFV-based FHE schemes, which are
optimized for integer arithmetic without rescaling requirements.

## Input/Output

- **Input**: MLIR module with `secret` dialect operations annotated with
  `mgmt.mgmt` attributes indicating ciphertext levels and management operations
- **Output**: Same module with `bgv.schemeParam` attribute added at module level
  containing generated BFV parameters (logN, Q, P, plaintextModulus)

## Options

- `--model` (string, default: "bfv-noise-kpz21"): Noise model to use for
  parameter generation
  - `bfv-noise-by-bound-coeff-average-case`: Uses coefficient bound model with
    2√N expansion factor
  - `bfv-noise-by-bound-coeff-worst-case` or `bfv-noise-kpz21`: Uses KPZ21
    coefficient bound model with N expansion factor
  - `bfv-noise-by-variance-coeff` or `bfv-noise-bmcm23`: Uses BMCM23 variance
    tracking model for independent ciphertext inputs
- `--mod-bits` (int, default: 60): Number of bits for prime coefficient modulus
  (use 57 for smaller machine words)
- `--slot-number` (int, default: 0): Minimum number of slots required for
  parameter generation
- `--plaintext-modulus` (int64, default: 65537): Plaintext modulus for the BFV
  scheme
- `--use-public-key` (bool, default: true): Whether to use public key encryption
  (affects security parameters)
- `--encryption-technique-extended` (bool, default: false): Whether to use
  extended encryption technique

## Usage Examples

```bash
# Generate BFV parameters with default settings
heir-opt --generate-param-bfv input.mlir

# Use variance-based noise model for better accuracy
heir-opt --generate-param-bfv --model=bfv-noise-bmcm23 input.mlir

# Generate parameters with smaller modulus bits
heir-opt --generate-param-bfv --mod-bits=57 input.mlir

# Use specific plaintext modulus and slot count
heir-opt --generate-param-bfv --plaintext-modulus=1024 --slot-number=2048 input.mlir

# Use secret key encryption only
heir-opt --generate-param-bfv --use-public-key=false input.mlir
```

## When to Use

This pass should be used in BFV-based FHE compilation pipelines after:

1. Circuit has been converted to `secret` dialect operations
1. Management operations (`mgmt.relinearize`) have been inserted
1. Operations have been annotated with `mgmt.mgmt` attributes (via
   `--annotate-mgmt`)

BFV is particularly suitable for applications requiring:

- Exact integer arithmetic
- No rescaling operations
- Simpler noise growth patterns compared to BGV

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
module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 12, Q = [1152921504606994433], P = [1152921504607191041], plaintextModulus = 65537>} {
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
