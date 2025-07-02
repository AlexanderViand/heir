---
title: ckks-to-lwe
weight: 21
---

## Overview

Converts CKKS dialect operations to the generic LWE dialect, providing a
backend-agnostic intermediate representation for approximate arithmetic
homomorphic encryption operations. This pass abstracts away CKKS-specific
details while preserving essential floating-point FHE semantics, enabling common
optimizations and multiple backend targets.

## Input/Output

- **Input**: IR with CKKS dialect operations (`ckks.add`, `ckks.mul`,
  `ckks.rescale`, etc.)
- **Output**: IR with generic LWE dialect operations that maintain approximate
  arithmetic semantics without scheme-specific details

## Options

This pass does not currently expose command-line options.

## Usage Examples

```bash
# Basic usage
heir-opt --ckks-to-lwe input.mlir

# Common pipeline: convert secret to CKKS, then to LWE
heir-opt --secret-to-ckks --ckks-to-lwe input.mlir

# Full pipeline to backend for floating-point FHE
heir-opt --secret-to-ckks --ckks-to-lwe --lwe-to-openfhe input.mlir
```

## When to Use

- **Backend Abstraction**: When targeting multiple FHE libraries with CKKS
  support (OpenFHE, Lattigo, etc.)
- **Cross-Scheme Optimization**: To apply common optimization passes for
  approximate arithmetic FHE
- **Analysis Passes**: Before running LWE-level analysis on floating-point FHE
  operations
- **Pipeline Flexibility**: To enable switching between backends without
  changing upstream floating-point passes

## Prerequisites

- Input must contain CKKS dialect operations
- Should run after `secret-to-ckks` or equivalent CKKS generation passes
- Compatible with tensor-based SIMD operations for floating-point data

## Technical Notes

- **Approximate Arithmetic**: Preserves CKKS's approximate arithmetic semantics
  at the LWE level
- **Scale Management**: Maintains scale factors and precision control in the
  abstracted representation
- **Tensor Support**: Handles tensor operations for SIMD-style floating-point
  computation
- **Partial Conversion**: Some CKKS-specific operations (e.g., `ckks.rescale`)
  may remain unchanged if they have no direct LWE analogue

## Limitations

- Currently implements "common" lowering mode; "full" lowering mode planned
  (TODO #1193)
- Some CKKS-specific optimizations may be lost during abstraction
- Complex number semantics may be simplified during conversion
- CKKS-specific operations without LWE equivalents remain unchanged

## Related Passes

- `secret-to-ckks`: Generates CKKS operations (should run before)
- `lwe-to-openfhe`: Backend targeting for OpenFHE (typically runs after)
- `lwe-to-lattigo`: Backend targeting for Lattigo (typically runs after)
- `bgv-to-lwe`: Similar abstraction for exact arithmetic operations
- `populate-scale-ckks`: Scale management for CKKS operations (may run before)
