---
title: Fold Constant Tensors
weight: 13
---

## Overview

The `fold-constant-tensors` pass performs compile-time evaluation of tensor
operations when all operands are constants. This specialized folding pass
complements general constant folding by handling tensor-specific operations that
may not be covered by standard canonicalization.

## Input/Output

- **Input**: MLIR IR containing tensor operations with constant operands
- **Output**: MLIR IR with constant tensor operations folded into new constant
  values

## Options

This pass takes no command-line options.

## Usage Examples

```bash
# Fold constant tensor operations
heir-opt --fold-constant-tensors input.mlir

# Common usage in optimization pipeline
heir-opt --full-loop-unroll --forward-insert-to-extract --fold-constant-tensors input.mlir

# With comprehensive constant folding
heir-opt --fold-constant-tensors --apply-folders input.mlir
```

## When to Use

This pass is beneficial when:

1. **After tensor forwarding**: Following `forward-insert-to-extract` when
   constant tensors become apparent
1. **Constant tensor optimization**: When working with tensors that have known
   constant values
1. **FHE preprocessing**: Before lowering to FHE schemes where constant
   evaluation reduces circuit complexity
1. **Compile-time optimization**: To reduce runtime overhead by precomputing
   tensor operations
1. **Shape optimization**: When tensor shape transformations can be evaluated at
   compile time

The pass is typically used after other optimization passes that expose constant
tensor operations.

## Implementation Details

**Supported Operations:**

- **`tensor.insert`**: Folding insertion of constants into constant tensors
- **`tensor.collapse_shape`**: Folding shape transformations of constant tensors

The transformation process:

1. **Constant Detection**: Identifies tensor operations where all operands are
   compile-time constants
1. **Operation Evaluation**: Evaluates supported tensor operations at compile
   time
1. **Constant Replacement**: Replaces the operation with a new constant tensor
   result
1. **Optimization Propagation**: Creates opportunities for further constant
   folding in dependent operations

**Benefits:**

- Eliminates runtime tensor operations when possible
- Reduces memory allocation for intermediate constant tensors
- Simplifies IR for subsequent optimization passes
- Improves performance by moving computation from runtime to compile time
- Enables better analysis and optimization of tensor-heavy code

**Limitations:**

- Only handles operations where all operands are constants
- Limited to specific tensor operations (insert, collapse_shape)
- Does not perform symbolic or partial evaluation

**Pipeline Integration:**

- Typically used after passes that expose constant values
- Complements `apply-folders` with tensor-specific optimizations
- Often followed by dead code elimination to remove unused operations
