---
title: secret-to-ckks
weight: 11
---

## Overview

Converts `secret.generic` operations containing arithmetic operations to CKKS
(Cheon-Kim-Kim-Song) ciphertext operations. This pass enables approximate
homomorphic computation on encrypted floating-point data, making it ideal for
machine learning inference and other applications requiring real number
arithmetic.

## Input/Output

- **Input**: IR with `secret.generic` blocks containing arithmetic operations on
  secret values
- **Output**: IR with CKKS dialect operations (`ckks.add`, `ckks.mul`, etc.)
  operating on encrypted ciphertexts with approximate arithmetic

## Options

- `--poly-mod-degree`: Default degree of the cyclotomic polynomial modulus for
  ciphertext space (default: 1024)

## Usage Examples

```bash
# Basic usage with default polynomial degree
heir-opt --secret-to-ckks input.mlir

# Specify custom polynomial degree
heir-opt --secret-to-ckks=poly-mod-degree=2048 input.mlir

# Common pipeline: distribute generics first, then convert
heir-opt --secret-distribute-generic --canonicalize --secret-to-ckks input.mlir
```

## When to Use

- When targeting CKKS-based FHE backends for approximate arithmetic
- For machine learning inference workloads on encrypted data
- When working with floating-point computations that can tolerate precision loss
- For applications requiring SIMD-packed operations on real numbers
- Before lowering to LWE or backend-specific dialects

## Prerequisites

- Input must contain properly distributed `secret.generic` blocks
- Should run `canonicalize` beforehand to remove non-secret values from block
  arguments
- All tensor types must have uniform shapes matching the ciphertext space
  dimension
- Floating-point operations should be designed to handle controlled precision
  loss

## Technical Notes

- **Approximate Arithmetic**: Unlike BGV, CKKS provides approximate results with
  controlled precision loss
- **Scale Management**: Automatically handles scale factors for fixed-point
  encoding of real numbers
- **SIMD Support**: Supports vectorized operations through complex number
  packing
- **Noise Growth**: Manages noise accumulation through rescaling operations

## Related Passes

- `secret-distribute-generic`: Distributes secret operations (should run before)
- `ckks-to-lwe`: Lowers CKKS to LWE dialect (typically runs after)
- `secret-to-bgv`: Alternative for exact integer arithmetic
- `populate-scale-ckks`: Sets scale values for CKKS operations
