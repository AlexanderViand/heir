---
title: Drop Unit Dims
weight: 15
---

## Overview

The `drop-unit-dims` pass optimizes linalg operations by converting them to
specialized variants that eliminate unit dimensions. This transformation
improves performance by using more efficient operation types when tensors have
dimensions of size 1.

## Input/Output

- **Input**: MLIR IR containing linalg operations with operands that have unit
  dimensions
- **Output**: MLIR IR with specialized linalg operations that operate on
  reduced-dimension tensors

## Options

This pass takes no command-line options.

## Usage Examples

```bash
# Drop unit dimensions from linalg operations
heir-opt --drop-unit-dims input.mlir

# Common usage in tensor optimization pipeline
heir-opt --convert-elementwise-to-affine --drop-unit-dims input.mlir
```

## When to Use

This pass is beneficial when:

1. **Matrix-vector optimization**: When matrix operations have one dimension of
   size 1, converting them to more efficient vector operations
1. **FHE computation optimization**: In FHE contexts where matrix-vector
   operations are common and performance-critical
1. **Before linalg lowering**: To use specialized kernels for lower-dimensional
   operations
1. **Memory optimization**: To reduce computational complexity by eliminating
   trivial dimensions
1. **Kernel specialization**: When specialized implementations exist for
   reduced-dimension operations

## Implementation Details

**Common Transformations:**

- **`linalg.matmul`** with RHS of shape `NxMx1` → **`linalg.matvec`** with
  shapes `NxM` and `M`
- Similar optimizations for other linalg operations with unit dimensions

The transformation process:

1. **Dimension Analysis**: Identifies linalg operations with operands containing
   unit dimensions
1. **Operation Specialization**: Selects appropriate specialized linalg
   operations for reduced dimensions
1. **Type Adaptation**: Adjusts tensor types to remove unit dimensions
1. **Semantic Preservation**: Ensures mathematical equivalence while improving
   efficiency

**Benefits:**

- Reduces computational complexity for operations with unit dimensions
- Enables use of specialized, more efficient operation implementations
- Improves memory access patterns by eliminating trivial dimensions
- Provides better optimization opportunities for subsequent passes

**Supported Operations:**

- Matrix multiplication operations where one operand has unit dimensions
- Other linalg operations that have specialized variants for reduced dimensions

**Limitations:**

- Only applies to operations where specialized unit-dimension variants exist
- Requires static knowledge of tensor shapes to identify unit dimensions
