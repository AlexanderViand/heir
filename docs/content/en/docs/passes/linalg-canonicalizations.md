---
title: Linalg Canonicalization
weight: 18
---

## Overview

The `linalg-canonicalizations` pass performs compile-time canonicalization of
linalg operations, specifically optimizing transpose operations on constant
tensors by evaluating them at compile time. This transformation reduces runtime
overhead and enables better optimization opportunities.

## Input/Output

- **Input**: MLIR IR containing `linalg.transpose` operations on constant
  tensors
- **Output**: MLIR IR with transpose operations replaced by pre-computed
  transposed constants

## Options

This pass takes no command-line options.

## Usage Examples

```bash
# Canonicalize linalg operations including constant transposes
heir-opt --linalg-canonicalizations input.mlir

# Common usage in tensor optimization pipeline
heir-opt --fold-constant-tensors --linalg-canonicalizations input.mlir
```

## When to Use

This pass is beneficial when:

1. **Constant tensor optimization**: When working with known constant matrices
   that undergo transpose operations
1. **Preprocessing for FHE**: Before lowering to FHE schemes where runtime
   transpose operations are expensive
1. **Compile-time evaluation**: To move computations from runtime to compile
   time when possible
1. **Memory optimization**: To reduce intermediate tensor allocations for
   transpose operations
1. **Pipeline optimization**: As part of a broader constant folding and
   canonicalization strategy

The pass is typically used alongside other constant folding passes for
comprehensive optimization.

## Implementation Details

**Supported Canonicalization:**

- **Constant Transpose**: `linalg.transpose` operations on compile-time constant
  tensors

The transformation process:

1. **Constant Detection**: Identifies `linalg.transpose` operations where the
   input tensor is a compile-time constant
1. **Transpose Evaluation**: Performs the transpose operation at compile time
1. **Constant Replacement**: Replaces the operation with a new constant tensor
   containing the transposed data
1. **Optimization Propagation**: Creates opportunities for further constant
   folding in dependent operations

**Benefits:**

- Eliminates runtime transpose operations on known constants
- Reduces memory allocation for intermediate transpose results
- Improves performance by moving computation to compile time
- Simplifies IR for subsequent optimization passes
- Enables better analysis of tensor access patterns

**Example Transformation:**

```mlir
// Input: Runtime transpose of constant
%const = arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
%transposed = linalg.transpose ins(%const : tensor<2x2xi32>)
                              outs(%output : tensor<2x2xi32>)
                              permutation = [1, 0]

// Output: Pre-computed transposed constant
%transposed = arith.constant dense<[[1, 3], [2, 4]]> : tensor<2x2xi32>
```

**Pipeline Integration:**

- Works well with other constant folding passes
- Typically used after tensor creation but before lowering
- Complements general canonicalization with linalg-specific optimizations
