---
title: Unroll and Forward
weight: 14
---

## Overview

The `unroll-and-forward` pass combines loop unrolling with store-to-load
forwarding to eliminate memory operations in favor of direct value usage. This
optimization is crucial for FHE compilation where memory operations are
expensive and should be minimized in favor of register-based computation.

## Input/Output

- **Input**: Function with affine loops containing load/store operations on
  memrefs
- **Output**: Same function with loops unrolled and memory accesses replaced by
  direct value forwarding

## Options

This pass takes no options - it processes the first function in the module and
applies transformations automatically.

## Usage Examples

```bash
# Apply unrolling and forwarding to eliminate memory operations
heir-opt --unroll-and-forward input.mlir

# Use in a typical memref elimination pipeline
heir-opt --expand-copy --unroll-and-forward --memref-global-replace input.mlir
```

## When to Use

This pass should be used in compilation pipelines that need to:

1. **Eliminate Memory Operations**: Convert memref-based code to register-only
   operations
1. **Prepare for FHE**: Remove memory allocations that cannot be handled in
   encrypted domains
1. **Optimize Loop-Based Computations**: Transform iterative algorithms into
   straight-line code

Common pipeline position:

```bash
heir-opt --tensor-to-memref --expand-copy --unroll-and-forward --memref-global-replace
```

## Algorithm

The pass performs the following steps iteratively:

1. **Loop Unrolling**: Fully unroll the first loop in the function
1. **Load Analysis**: Find all load operations with statically-inferrable
   indices
1. **Store Tracking**: For each load, backtrack to find all corresponding stores
1. **Value Forwarding**: Replace loads with the values from the last
   corresponding store
1. **Memref Argument Handling**: Forward through renames/subviews to original
   function arguments
1. **Remaining Loads**: Apply same logic to loads not inside loops

## Memory Operation Requirements

- **Static Indices**: Only handles loads/stores with statically-determinable
  indices
- **Affine Operations**: Requires affine.load and affine.store operations
- **Memref Allocation**: Tracks back to original memref.alloc operations
- **Exclude Globals**: Global memrefs are handled by separate
  `memref-global-replace` pass

## Example Transformation

**Input:**

```mlir
func.func @compute_with_loops() -> i32 {
  %alloc = memref.alloc() : memref<4xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %init = arith.constant 10 : i32

  // Initialize array
  affine.for %i = 0 to 4 {
    affine.store %init, %alloc[%i] : memref<4xi32>
  }

  // Compute sum
  %sum_init = arith.constant 0 : i32
  %sum_alloc = memref.alloc() : memref<1xi32>
  affine.store %sum_init, %sum_alloc[%c0] : memref<1xi32>

  affine.for %i = 0 to 4 {
    %current_sum = affine.load %sum_alloc[%c0] : memref<1xi32>
    %value = affine.load %alloc[%i] : memref<4xi32>
    %new_sum = arith.addi %current_sum, %value : i32
    affine.store %new_sum, %sum_alloc[%c0] : memref<1xi32>
  }

  %result = affine.load %sum_alloc[%c0] : memref<1xi32>
  return %result : i32
}
```

**Output:**

```mlir
func.func @compute_with_loops() -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %init = arith.constant 10 : i32
  %sum_init = arith.constant 0 : i32

  // First loop unrolled - stores forwarded away
  // (no operations needed, stores were eliminated)

  // Second loop unrolled with forwarding
  // Iteration 0
  %sum_0 = arith.addi %sum_init, %init : i32  // forwarded values

  // Iteration 1
  %sum_1 = arith.addi %sum_0, %init : i32

  // Iteration 2
  %sum_2 = arith.addi %sum_1, %init : i32

  // Iteration 3
  %sum_3 = arith.addi %sum_2, %init : i32

  // Final result forwarded
  return %sum_3 : i32
}
```

## Forwarding Analysis

The pass performs sophisticated analysis to:

### Store-Load Matching

- Match loads to their corresponding stores based on memref and indices
- Handle multiple stores to the same location (last store wins)
- Track stores through control flow within loops

### Memref Tracking

- Follow memref aliases through subview operations
- Handle memref.cast operations
- Trace back to original allocation or function arguments

### Value Propagation

- Forward stored values directly to load sites
- Eliminate intermediate memref operations
- Preserve computation dependencies

## Benefits for FHE

- **No Memory Allocation**: Eliminates memref.alloc operations that cannot be
  encrypted
- **Direct Value Flow**: Creates register-only computation suitable for FHE
- **Reduced Complexity**: Simplifies IR for subsequent FHE-specific
  optimizations
- **Better Analysis**: Enables more precise dataflow analysis for encrypted
  computations

## Prerequisites

- Input must use affine dialect for memory operations
- Tensor operations should be lowered to memref first
- Static loop bounds and array access patterns required
- Should run after `expand-copy` if memref.copy operations are present
