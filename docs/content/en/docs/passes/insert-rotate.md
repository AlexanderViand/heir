---
title: Insert Rotate
weight: 4
---

## Overview

The insert rotate pass implements SIMD-vectorization using HECO-style heuristics
for FHE operations. This pass identifies arithmetic operations that can be
combined into cyclic rotations and vectorized tensor operations, implementing
the automatic SIMD batching approach from the
[HECO paper](https://arxiv.org/abs/2202.01649).

## Input/Output

- **Input**: IR with plaintext tensor operations and arithmetic operations that
  can be vectorized
- **Output**: IR with `tensor_ext.rotate` operations and vectorized arithmetic
  operations, optimized for SIMD execution

## Options

This pass has no command-line options.

## Usage Examples

```bash
heir-opt --insert-rotate input.mlir
```

For optimal results, combine with cleanup passes:

```bash
heir-opt --insert-rotate --cse --canonicalize --collapse-insertion-chains input.mlir
```

## When to Use

Use this pass in FHE compilation pipelines for:

1. Vectorizing unrolled affine loops
1. Converting scalar arithmetic to SIMD operations
1. Optimizing plaintext operations before scheme dialect lowering

Typical placement:

1. After straight-line vectorization
1. Before `collapse-insertion-chains` cleanup
1. Before lowering to FHE scheme dialects (e.g., `bgv`)
1. Should be followed by `--cse` and `--canonicalize`

## How It Works

The pass operates by:

1. Identifying arithmetic operations that extract scalar values from tensors
1. Analyzing target slots for each operation using heuristics
1. Inserting `tensor_ext.rotate` operations to align data in SIMD slots
1. Replacing extract-operate patterns with vectorized equivalents
1. Optimizing rotation patterns to minimize total rotation count

The implementation includes patterns for:

- Two-tensor arithmetic operations
- Scalar-tensor operations (via splatting)
- Rotation alignment optimizations

## Example

**Before insert-rotate:**

```mlir
func.func @vectorizable_sum(%t: tensor<8xi32>) -> i32 {
  %0 = tensor.extract %t[0] : tensor<8xi32>
  %1 = tensor.extract %t[1] : tensor<8xi32>
  %2 = tensor.extract %t[2] : tensor<8xi32>
  %3 = tensor.extract %t[3] : tensor<8xi32>
  %4 = arith.addi %0, %1 : i32
  %5 = arith.addi %4, %2 : i32
  %6 = arith.addi %5, %3 : i32
  return %6 : i32
}
```

**After insert-rotate:**

```mlir
func.func @vectorizable_sum(%t: tensor<8xi32>) -> i32 {
  %r1 = tensor_ext.rotate %t, 1 : tensor<8xi32>
  %add1 = arith.addi %t, %r1 : tensor<8xi32>
  %r2 = tensor_ext.rotate %add1, 2 : tensor<8xi32>
  %add2 = arith.addi %add1, %r2 : tensor<8xi32>
  %result = tensor.extract %add2[0] : tensor<8xi32>
  return %result : i32
}
```

## Supported Operations

The pass supports vectorization of:

- `arith.addi`, `arith.subi`, `arith.muli` (integer arithmetic)
- `arith.addf`, `arith.subf`, `arith.mulf` (floating-point arithmetic)
- Operations with constant operands (via splatting)
- Nested arithmetic expressions

## Cleanup Requirements

This pass by itself does not eliminate operations but inserts well-chosen
rotations. For optimal results:

1. **Required**: Run `--cse` and `--canonicalize` after this pass
1. **Recommended**: Follow with `--collapse-insertion-chains`
1. These cleanup passes dramatically reduce the IR for well-structured code

## Algorithmic Details

The pass implements target slot analysis to:

- Determine optimal alignment for each operation
- Minimize unnecessary rotations
- Handle left and right associative groupings
- Align rotation patterns across operation chains

## Related Passes

- **Prerequisite**: `straight-line-vectorize` for initial vectorization
- **Cleanup**: `collapse-insertion-chains` for cleaning up insertion patterns
- **Essential**: `--cse` and `--canonicalize` for IR simplification
- **Pipeline**: Should be used before lowering to scheme dialects
