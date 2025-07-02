---
title: secret-to-bgv
weight: 10
---

## Overview

Converts `secret.generic` operations containing arithmetic operations to BGV
(Brakerski-Gentry-Vaikuntanathan) ciphertext operations. This pass is a key
component in the FHE compilation pipeline for schemes that use the BGV
homomorphic encryption scheme.

## Input/Output

- **Input**: IR with `secret.generic` blocks containing arithmetic operations on
  secret values
- **Output**: IR with BGV dialect operations (`bgv.add`, `bgv.mul`, etc.)
  operating on encrypted ciphertexts

## Options

- `--poly-mod-degree`: Default degree of the cyclotomic polynomial modulus for
  ciphertext space (default: 1024)

## Usage Examples

```bash
# Basic usage with default polynomial degree
heir-opt --secret-to-bgv input.mlir

# Specify custom polynomial degree
heir-opt --secret-to-bgv=poly-mod-degree=2048 input.mlir

# Common pipeline: distribute generics first, then convert
heir-opt --secret-distribute-generic --canonicalize --secret-to-bgv input.mlir
```

## When to Use

- When targeting BGV-based FHE backends (e.g., SEAL, OpenFHE with BGV)
- For applications requiring exact arithmetic over integers
- After secret values have been distributed through arithmetic operations
- Before lowering to LWE or backend-specific dialects

## Prerequisites

- Input must contain properly distributed `secret.generic` blocks
- Should run `canonicalize` beforehand to remove non-secret values from block
  arguments
- All tensor types must have uniform shapes matching the ciphertext space
  dimension

## Related Passes

- `secret-distribute-generic`: Distributes secret operations (should run before)
- `bgv-to-lwe`: Lowers BGV to LWE dialect (typically runs after)
- `secret-to-ckks`: Alternative for floating-point computations
