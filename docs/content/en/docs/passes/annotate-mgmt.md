---
title: Annotate Management
weight: 12
---

## Overview

The `annotate-mgmt` pass performs secretness, level, and dimension analysis on
the IR and annotates operations with management information. This pass is
crucial for FHE parameter generation and optimization, as it determines
ciphertext properties throughout the computation.

## Input/Output

- **Input**: MLIR module with secret dialect operations
- **Output**: Same module with `mgmt.mgmt` attributes added to operations,
  containing analysis results for secretness, ciphertext levels, and dimensions

## Options

- `--base-level` (int, default: 0): Starting level for ciphertext level counting
  - Used by BGV/BFV schemes where level counting may start from a different base
  - CKKS typically uses level 0 as the base

## Usage Examples

```bash
# Annotate management attributes with default base level
heir-opt --annotate-mgmt input.mlir

# Use specific base level for BGV/BFV schemes
heir-opt --annotate-mgmt --base-level=1 input.mlir

# Annotate management for CKKS (typically uses default base level 0)
heir-opt --annotate-mgmt --base-level=0 input.mlir
```

## When to Use

This pass is essential in FHE compilation pipelines and should be used:

1. **Before Parameter Generation**: Required by `generate-param-*` passes to
   understand circuit requirements
1. **After Management Insertion**: Run after passes like `secret-insert-mgmt-*`
   that insert management operations
1. **For Optimization Passes**: Enables other passes like
   `optimize-relinearization` to make informed decisions

Typical pipeline position:

```bash
heir-opt --secret-insert-mgmt-bgv --annotate-mgmt --generate-param-bgv input.mlir
```

## Analysis Components

The pass performs three types of analysis:

### 1. Secretness Analysis

- Determines which values are encrypted (secret) vs. plaintext (public)
- Propagates secretness through operations
- Critical for determining which operations require homomorphic computation

### 2. Level Analysis

- Tracks ciphertext multiplication depth/level
- Essential for determining modulus consumption in leveled FHE schemes
- Used by parameter generation to size modulus chains appropriately

### 3. Dimension Analysis

- Tracks ciphertext dimensions for schemes supporting packed operations
- Important for SIMD-style computations in FHE
- Helps optimize slot utilization

## Management Attribute Format

The pass adds `mgmt.mgmt` attributes with the following information:

```mlir
{mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = [1024], secretness = secret>}
```

- **level**: Ciphertext multiplication depth
- **dimension**: Vector of dimension sizes for packed ciphertexts
- **secretness**: Whether the value is secret (encrypted) or public (plaintext)

## Example Transformation

**Input:**

```mlir
module {
  func.func @compute(%arg0: !secret.secret<i16>, %arg1: !secret.secret<i16>) -> !secret.secret<i16> {
    %0 = secret.generic(%arg0, %arg1 : !secret.secret<i16>, !secret.secret<i16>) {
    ^body(%input0: i16, %input1: i16):
      %1 = arith.addi %input0, %input1 : i16
      %2 = arith.muli %1, %input1 : i16
      secret.yield %2 : i16
    } -> !secret.secret<i16>
    return %0 : !secret.secret<i16>
  }
}
```

**Output:**

```mlir
module {
  func.func @compute(%arg0: !secret.secret<i16>, %arg1: !secret.secret<i16>) -> !secret.secret<i16> {
    %0 = secret.generic(%arg0, %arg1 : !secret.secret<i16>, !secret.secret<i16>)
         attrs = {arg0 = {mgmt.mgmt = #mgmt.mgmt<level = 0>},
                  arg1 = {mgmt.mgmt = #mgmt.mgmt<level = 0>}} {
    ^body(%input0: i16, %input1: i16):
      %1 = arith.addi %input0, %input1 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : i16
      %2 = arith.muli %1, %input1 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : i16
      secret.yield %2 : i16
    } -> !secret.secret<i16>
    return %0 : !secret.secret<i16>
  }
}
```

## Analysis Accuracy

The analysis makes conservative assumptions:

- **Secretness**: Errs on the side of treating values as secret when uncertain
- **Levels**: Tracks worst-case multiplication depth
- **Dimensions**: Handles dynamic dimensions conservatively

## Integration with Other Passes

- **Parameter Generation**: `generate-param-*` passes read these annotations
- **Optimization**: `optimize-relinearization` uses level information
- **Validation**: `validate-noise` uses annotations for noise analysis
- **Backend Lowering**: Target-specific passes use management information for
  code generation
