---
title: Convert to Ciphertext Semantics
weight: 4
---

## Overview

The `convert-to-ciphertext-semantics` pass converts programs from tensor
semantics to ciphertext semantics, transforming how tensor operations are
interpreted in the context of FHE computations. This pass also implements
high-level tensor operations in terms of SIMD-style operations suitable for FHE.

## Input/Output

- **Input**: MLIR IR with tensor semantics (standard tensor operations)
- **Output**: MLIR IR with ciphertext semantics where tensors represent FHE
  ciphertext slots

## Semantic Transformation

### Tensor Semantics → Ciphertext Semantics

**Tensor Semantics**: Tensor-typed values are manipulated according to standard
MLIR tensor operations.

**Ciphertext Semantics**: Tensor-typed values correspond to tensors of FHE
ciphertexts, where the last dimension represents the number of ciphertext slots.

### Examples

- `tensor<32x32xi16>` (tensor semantics) → `tensor<65536xi16>` (ciphertext
  semantics)
- `tensor<64x64xi16>` (tensor semantics) → `tensor<4x32768xi16>` (ciphertext
  semantics)

## Options

| Option            | Type | Default | Description                                                  |
| ----------------- | ---- | ------- | ------------------------------------------------------------ |
| `ciphertext-size` | int  | `1024`  | Power of two length of the ciphertexts the data is packed in |

## Usage Examples

```bash
# Convert with default ciphertext size
heir-opt --convert-to-ciphertext-semantics input.mlir

# Convert with custom ciphertext size
heir-opt --convert-to-ciphertext-semantics='ciphertext-size=4096' input.mlir
```

## When to Use

This pass is essential when:

1. **FHE lowering**: Transitioning from high-level tensor operations to
   FHE-specific operations
1. **SIMD optimization**: Preparing for SIMD-style operations on packed
   ciphertexts
1. **Layout optimization**: After layout assignment but before concrete FHE
   dialect lowering
1. **Kernel implementation**: When high-level operations like `linalg.matvec`
   need FHE implementations

## Key Features

### Dual Role

1. **Type Conversion**: Transforms tensor types to ciphertext-semantic tensor
   types
1. **Operation Implementation**: Implements high-level tensor ops (e.g.,
   `linalg.matvec`) as SIMD/rotation operations

### Metadata Preservation

- Function arguments and return values are annotated with `secret.original_type`
- Enables later passes to implement proper encoding/decoding for specific FHE
  schemes

### Intermediate Representation

Ciphertext semantics serve as an intermediate step between:

- High-level tensor operations
- Concrete FHE scheme dialects (e.g., `lwe`)

This enables:

- Generic FHE optimizations
- Scheme-agnostic transformations
- Easier implementation of cross-scheme optimizations

## Implementation Details

The pass handles:

- Tensor type conversion based on ciphertext slot capacity
- High-level operation lowering to SIMD-style operations
- Original type preservation for encoding/decoding
- Integration with layout analysis and assignment
