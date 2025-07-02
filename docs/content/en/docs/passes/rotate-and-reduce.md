---
title: Rotate and Reduce
weight: 30
---

## Overview

The rotate-and-reduce pass optimizes reduction operations on tensors by using a
logarithmic number of rotation operations instead of linear sequential
extractions. This is particularly beneficial for FHE schemes that support SIMD
operations, as it significantly reduces the total number of operations required
for tensor reductions.

## Input/Output

- **Input**: IR containing unrolled reduction patterns with sequential tensor
  extractions and binary operations
- **Output**: Optimized IR using `tensor_ext.rotate` operations to perform
  reductions in logarithmic steps

## Options

This pass has no command-line options.

## Usage Examples

```bash
heir-opt --rotate-and-reduce input.mlir
```

Typically used in combination with other tensor optimization passes:

```bash
heir-opt --insert-rotate --rotate-and-reduce --canonicalize input.mlir
```

## When to Use

Use this pass when you have:

1. Commutative and associative binary operations reducing all tensor entries to
   a single value
1. Unrolled tensor extraction patterns (often from loop unrolling)
1. FHE schemes that benefit from rotation-based operations (like BGV, CKKS)
1. IR where reduction operations can be optimized using SIMD parallelism

Typical placement in compilation pipelines:

1. After loop unrolling passes that create extraction patterns
1. After `insert-rotate` which prepares the IR for rotation optimizations
1. Before lowering to scheme-specific dialects
1. Often followed by `--canonicalize` to clean up the result

## How It Works

The pass identifies unrolled reduction patterns and replaces them with a tree of
operations using logarithmic depth:

1. Detects complete tensor reductions with commutative/associative operations
1. Validates that all tensor entries are being reduced
1. Replaces the linear reduction chain with rotation-based operations
1. Uses powers-of-2 rotation distances for optimal tree structure

## Example

**Before optimization:**

```mlir
%0 = tensor.extract %t[0] : tensor<8xi32>
%1 = tensor.extract %t[1] : tensor<8xi32>
%2 = tensor.extract %t[2] : tensor<8xi32>
%3 = tensor.extract %t[3] : tensor<8xi32>
%4 = tensor.extract %t[4] : tensor<8xi32>
%5 = tensor.extract %t[5] : tensor<8xi32>
%6 = tensor.extract %t[6] : tensor<8xi32>
%7 = tensor.extract %t[7] : tensor<8xi32>
%8 = arith.addi %0, %1 : i32
%9 = arith.addi %8, %2 : i32
%10 = arith.addi %9, %3 : i32
%11 = arith.addi %10, %4 : i32
%12 = arith.addi %11, %5 : i32
%13 = arith.addi %12, %6 : i32
%14 = arith.addi %13, %7 : i32
```

**After optimization:**

```mlir
%0 = tensor_ext.rotate %t, 4 : tensor<8xi32>
%1 = arith.addi %t, %0 : tensor<8xi32>
%2 = tensor_ext.rotate %1, 2 : tensor<8xi32>
%3 = arith.addi %1, %2 : tensor<8xi32>
%4 = tensor_ext.rotate %3, 1 : tensor<8xi32>
%5 = arith.addi %3, %4 : tensor<8xi32>
```

## Benefits

- **Reduced Operation Count**: Converts O(n) operations to O(log n)
- **Improved Parallelism**: Enables SIMD execution of reduction operations
- **Lower Latency**: Logarithmic depth reduces critical path length
- **FHE Optimization**: Leverages native rotation capabilities of FHE schemes

## Limitations

- Only works with commutative and associative binary operations
- Requires unrolled tensor extractions (needs prior loop unrolling)
- Limited to complete tensor reductions (all elements must be reduced)
- Tensor size should be a power of 2 for optimal results

## Related Passes

- Use after `insert-rotate` for comprehensive rotation optimization
- Works well with `full-loop-unroll` to create suitable input patterns
- Combine with `--canonicalize` to clean up the result
- Often used with `collapse-insertion-chains` for complete vectorization
