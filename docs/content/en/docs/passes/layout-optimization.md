---
title: Layout Optimization
weight: 1
---

## Overview

The layout optimization pass performs a greedy optimization of layout
conversions in the IR to minimize the cost of ciphertext data movement
operations. This pass implements automatic layout assignment similar to the
approach from
[A Tensor Compiler with Automatic Data Packing for Simple and Efficient Fully Homomorphic Encryption](https://dl.acm.org/doi/pdf/10.1145/3656382).

## Input/Output

- **Input**: IR with initial layout assignments provided by the
  `layout-propagation` pass. Operations should be annotated with
  `tensor_ext.layout` attributes.
- **Output**: Optimized IR where layout conversions have been hoisted and
  consolidated to minimize overall conversion cost.

## Options

This pass has no command-line options.

## Usage Examples

```bash
heir-opt --layout-optimization input.mlir
```

For best results, combine with canonicalization:

```bash
heir-opt --layout-propagation --layout-optimization --canonicalize input.mlir
```

## When to Use

Use this pass after `layout-propagation` in compilation pipelines targeting FHE
schemes that support SIMD operations. It should be placed:

1. After `layout-propagation` has annotated operations with initial layouts
1. Before lowering to scheme-specific dialects (like `bgv` or `ckks`)
1. Typically followed by `--canonicalize` to clean up the optimized IR

## How It Works

The pass operates by:

1. Iterating through operations in reverse order
1. For each operation, attempting to hoist layout conversions of results before
   the operation
1. Computing the net cost of hoisting conversions considering:
   - Cost of performing the operation with new input layouts
   - Cost of converting input layouts
   - New cost of converting to other required output layouts
1. Selecting the layout conversion with the lowest net cost

## Example

**Before optimization:**

```mlir
%1 = tensor_ext.convert_layout %input1 {from_layout = #map1, to_layout = #map} : tensor<32xi16>
%2 = arith.addi %input0, %1 {tensor_ext.layout = #map} : tensor<32xi16>
%3 = tensor_ext.convert_layout %2 {from_layout = #map, to_layout = #map1} : tensor<32xi16>
%4 = arith.addi %3, %input2 {tensor_ext.layout = #map1} : tensor<32xi16>
```

**After optimization:**

```mlir
%1 = tensor_ext.convert_layout %input0 {from_layout = #map, to_layout = #map1} : tensor<32xi16>
%2 = arith.addi %1, %input1 {tensor_ext.layout = #map1} : tensor<32xi16>
%3 = arith.addi %2, %input2 {tensor_ext.layout = #map1} : tensor<32xi16>
```
