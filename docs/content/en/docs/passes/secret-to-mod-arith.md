---
title: secret-to-mod-arith
weight: 13
---

## Overview

Converts `secret.generic` operations containing arithmetic operations to modular
arithmetic operations using the `mod_arith` dialect. This pass enables plaintext
evaluation of FHE algorithms for testing, debugging, and prototyping purposes
without the overhead of encryption.

## Input/Output

- **Input**: IR with `secret.generic` blocks containing arithmetic operations on
  secret values
- **Output**: IR with `mod_arith` dialect operations performing modular
  arithmetic on plaintext values

## Options

- `--modulus`: Modulus to use for the mod-arith dialect (default: 0, uses
  natural modulus for integer type)
- `--log-scale`: Log base 2 of the scale for encoding floating points as
  integers (default: 0)

## Usage Examples

```bash
# Basic usage with automatic modulus selection
heir-opt --secret-to-mod-arith input.mlir

# Specify custom modulus
heir-opt --secret-to-mod-arith=modulus=65537 input.mlir

# Set scale for floating-point encoding
heir-opt --secret-to-mod-arith=log-scale=20 input.mlir

# Common pipeline: distribute generics first, then convert
heir-opt --secret-distribute-generic --canonicalize --secret-to-mod-arith input.mlir

# Floating-point with custom modulus and scale
heir-opt --secret-to-mod-arith=modulus=1073741824,log-scale=16 input.mlir
```

## When to Use

- **Algorithm Prototyping**: Testing FHE algorithms on plaintext data before
  applying encryption
- **Debugging**: Isolating logical errors from encryption-related issues
- **Performance Comparison**: Benchmarking plaintext vs. encrypted performance
- **Development**: Rapid iteration during FHE algorithm development
- **Testing**: Validating correctness of secret computations

## Prerequisites

- Input must contain properly distributed `secret.generic` blocks
- Should run `canonicalize` beforehand to remove non-secret values from block
  arguments
- All tensor types should be compatible with modular arithmetic operations

## Technical Notes

- **Automatic Modulus**: When modulus is not specified (default 0), uses the
  natural modulus for the integer type
- **Floating-Point Support**: Converts floating-point operations to fixed-point
  arithmetic using the log-scale parameter
- **Development Tool**: Primarily intended for development and testing, not
  production deployment
- **Performance**: Provides faster execution compared to encrypted alternatives

## Related Passes

- `secret-distribute-generic`: Distributes secret operations (should run before)
- `mod-arith-to-arith`: Converts back to standard arithmetic (may run after)
- `secret-to-bgv`: Alternative for encrypted BGV operations
- `secret-to-ckks`: Alternative for encrypted CKKS operations
