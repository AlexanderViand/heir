---
title: bgv-to-lwe
weight: 20
---

## Overview

Converts BGV dialect operations to the generic LWE dialect, providing a
backend-agnostic intermediate representation for homomorphic encryption
operations. This pass abstracts away BGV-specific details while preserving
essential FHE semantics, enabling common optimizations and multiple backend
targets.

## Input/Output

- **Input**: IR with BGV dialect operations (`bgv.add`, `bgv.mul`, etc.)
- **Output**: IR with generic LWE dialect operations (`lwe.add`, `lwe.mul`,
  etc.) that maintain FHE semantics without scheme-specific details

## Options

This pass does not currently expose command-line options.

## Usage Examples

```bash
# Basic usage
heir-opt --bgv-to-lwe input.mlir

# Common pipeline: convert secret to BGV, then to LWE
heir-opt --secret-to-bgv --bgv-to-lwe input.mlir

# Full pipeline to backend
heir-opt --secret-to-bgv --bgv-to-lwe --lwe-to-openfhe input.mlir
```

## When to Use

- **Backend Abstraction**: When targeting multiple FHE libraries (OpenFHE,
  Lattigo, etc.)
- **Cross-Scheme Optimization**: To apply common optimization passes across
  different FHE schemes
- **Analysis Passes**: Before running LWE-level analysis (noise, depth, etc.)
- **Pipeline Flexibility**: To enable switching between backends without
  changing upstream passes

## Prerequisites

- Input must contain BGV dialect operations
- Should run after `secret-to-bgv` or equivalent BGV generation passes
- Compatible with tensor-based SIMD operations

## Technical Notes

- **Abstraction Layer**: Removes BGV-specific implementation details while
  preserving FHE semantics
- **Preserved Semantics**: Maintains noise growth characteristics and operation
  semantics at the LWE level
- **Tensor Support**: Handles tensor operations for SIMD-style homomorphic
  computation
- **Partial Conversion**: Some BGV-specific operations (e.g., `bgv.modswitch`)
  may remain unchanged if they have no direct LWE analogue

## Limitations

- Currently implements "common" lowering mode; "full" lowering mode planned
  (TODO #1193)
- Some scheme-specific optimizations may be lost during abstraction
- BGV-specific operations without LWE equivalents remain unchanged

## Related Passes

- `secret-to-bgv`: Generates BGV operations (should run before)
- `lwe-to-openfhe`: Backend targeting for OpenFHE (typically runs after)
- `lwe-to-lattigo`: Backend targeting for Lattigo (typically runs after)
- `ckks-to-lwe`: Similar abstraction for CKKS operations
