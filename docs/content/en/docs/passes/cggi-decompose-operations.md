---
title: CGGI Decompose Operations
weight: 5
---

## Overview

The `cggi-decompose-operations` pass expands high-level CGGI operations (e.g.,
`cggi.lut2`, `cggi.xor`, `cggi.lut3`) into lower-level LWE operations and
programmable bootstrap operations. This transformation is essential for lowering
CGGI computations to the level where they can be executed by FHE libraries.

## Input/Output

- **Input**: MLIR module with high-level CGGI dialect operations (`cggi.xor`,
  `cggi.lut2`, `cggi.lut3`, `cggi.lut_lincomb`)
- **Output**: Module with operations expanded to LWE scalar operations and
  programmable bootstrap calls, or intermediate `cggi.lut_lincomb` operations
  depending on options

## Options

- `--expand-lincomb` (bool, default: true): Controls the level of expansion
  - `true`: Full expansion to LWE scalar operations and programmable bootstrap
    operations
  - `false`: Partial expansion stopping at `cggi.lut_lincomb` level

## Usage Examples

```bash
# Full expansion to LWE and PBS operations (default)
heir-opt --cggi-decompose-operations input.mlir

# Partial expansion to lut_lincomb level only
heir-opt --cggi-decompose-operations --expand-lincomb=false input.mlir
```

## When to Use

This pass should be used in CGGI compilation pipelines:

1. After high-level boolean operations have been converted to CGGI dialect
1. After parameters have been attached (via `--cggi-set-default-parameters` or
   proper parameter generation)
1. Before lowering to backend-specific operations (e.g., tfhe-rs, OpenFHE)

## Supported Operations

The pass supports decomposition of:

- **XOR operations**: Binary exclusive-or gates
- **LUT2 operations**: 2-input lookup table operations
- **LUT3 operations**: 3-input lookup table operations
- **LutLincomb operations**: Linear combination followed by lookup table

## Decomposition Strategy

For multi-input operations like LUT3, the decomposition follows this pattern:

1. **Linear combination**: Combine inputs using weighted sum (e.g., for inputs
   `c`, `b`, `a`: `4*c + 2*b + a`)
1. **Programmable bootstrap**: Apply the lookup table to the linear combination
   result

This approach leverages the structure of boolean functions where multi-bit
inputs can be encoded as single integers for table lookup.

## Example Transformation

**Input (LUT3 operation):**

```mlir
module {
  func.func @lut3_example(%c: !lwe.lwe_ciphertext, %b: !lwe.lwe_ciphertext, %a: !lwe.lwe_ciphertext) -> !lwe.lwe_ciphertext {
    %lut = arith.constant dense<[0, 1, 1, 0, 1, 0, 0, 1]> : tensor<8xi64>
    %0 = cggi.lut3 %c, %b, %a, %lut : !lwe.lwe_ciphertext
    return %0 : !lwe.lwe_ciphertext
  }
}
```

**Output (with expand-lincomb=true):**

```mlir
module {
  func.func @lut3_example(%c: !lwe.lwe_ciphertext, %b: !lwe.lwe_ciphertext, %a: !lwe.lwe_ciphertext) -> !lwe.lwe_ciphertext {
    %lut = arith.constant dense<[0, 1, 1, 0, 1, 0, 0, 1]> : tensor<8xi64>
    %c4 = arith.constant 4 : i64
    %c2 = arith.constant 2 : i64
    %c1 = arith.constant 1 : i64
    %scaled_c = lwe.mul_scalar %c, %c4 : (!lwe.lwe_ciphertext, i64) -> !lwe.lwe_ciphertext
    %scaled_b = lwe.mul_scalar %b, %c2 : (!lwe.lwe_ciphertext, i64) -> !lwe.lwe_ciphertext
    %scaled_a = lwe.mul_scalar %a, %c1 : (!lwe.lwe_ciphertext, i64) -> !lwe.lwe_ciphertext
    %sum1 = lwe.add %scaled_c, %scaled_b : !lwe.lwe_ciphertext
    %sum2 = lwe.add %sum1, %scaled_a : !lwe.lwe_ciphertext
    %0 = cggi.pbs %sum2, %lut : (!lwe.lwe_ciphertext, tensor<8xi64>) -> !lwe.lwe_ciphertext
    return %0 : !lwe.lwe_ciphertext
  }
}
```

## Relationship to Other Passes

- **Prerequisites**: Operations should have `cggi_params` attributes (from
  `--cggi-set-default-parameters`)
- **Follows**: High-level boolean circuit optimization passes
- **Precedes**: Backend-specific lowering passes (e.g., to tfhe-rs, OpenFHE)

## Performance Considerations

- Full expansion (`expand-lincomb=true`) provides maximum flexibility for
  subsequent optimization
- Partial expansion (`expand-lincomb=false`) may be preferred when targeting
  backends that can handle `lut_lincomb` operations directly
- The linear combination approach is efficient for TFHE-style bootstrapping
  where the cost is dominated by the PBS operation
