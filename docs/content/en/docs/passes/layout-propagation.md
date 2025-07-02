---
title: Layout Propagation
weight: 2
---

## Overview

The layout propagation pass performs a forward propagation of layout (packing)
information through the IR, establishing initial layout assignments for secret
tensor operations. This pass determines how plaintext data should be packed into
ciphertexts for SIMD operations.

## Input/Output

- **Input**: IR with secret tensor operations, typically from the `secret`
  dialect
- **Output**: IR annotated with layout information (`tensor_ext.layout`
  attributes) and necessary layout conversion operations

## Options

- `--ciphertext-size=<int>`: Power of two length of the ciphertexts the data is
  packed in (default: 1024)

## Usage Examples

```bash
heir-opt --layout-propagation input.mlir
```

With custom ciphertext size:

```bash
heir-opt --layout-propagation=ciphertext-size=2048 input.mlir
```

## When to Use

Use this pass early in the compilation pipeline for FHE schemes supporting SIMD
operations:

1. After secretization of the IR (when `secret.generic` operations are present)
1. Before layout optimization passes
1. As the first step in layout-aware transformations

## How It Works

The pass operates by:

1. Starting with the assumption that each secret tensor argument has a row-major
   layout
1. Propagating layout information forward through operations
1. Inserting `tensor_ext.assign_layout` ops when plaintext values are used with
   secret operations
1. Inserting `tensor_ext.convert_layout` ops when incompatible layouts are
   encountered
1. Annotating operations with layout attributes using:
   - `tensor_ext.layout` dialect attribute for function arguments
   - Affine map attributes for operation results

## Example

**Before propagation:**

```mlir
func.func @insert_conversion(%arg0: !stensor, %arg1: !stensor) -> !stensor2 {
  %out_1 = arith.constant dense<0> : !tensor2
  %out_2 = arith.constant dense<0> : !tensor2

  %0 = secret.generic(%arg0, %arg1: !stensor, !stensor) {
  ^body(%pt_arg0: !tensor, %pt_arg1: !tensor):
    %1 = linalg.reduce { arith.addi } ins(%pt_arg0:!tensor) outs(%out_1:!tensor2) dimensions = [0]
    %2 = linalg.reduce { arith.addi } ins(%pt_arg1:!tensor) outs(%out_2:!tensor2) dimensions = [1]
    %3 = arith.addi %1, %2 : !tensor2
    secret.yield %3 : !tensor2
  } -> !stensor2
  return %0 : !stensor2
}
```

**After propagation:**

```mlir
#map = affine_map<(d0, d1) -> (d0 * 32 + d1)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> (d0 * 32)>

func.func @insert_conversion(
  %arg0: !secret.secret<tensor<32x32xi16>> {tensor_ext.layout = #map},
  %arg1: !secret.secret<tensor<32x32xi16>> {tensor_ext.layout = #map}
) -> (!secret.secret<tensor<32xi16>> {tensor_ext.layout = #map1}) {
  %cst = arith.constant dense<0> : tensor<32xi16>
  %cst_0 = arith.constant dense<0> : tensor<32xi16>
  %0 = secret.generic(%arg0, %arg1 : !secret.secret<tensor<32x32xi16>>, !secret.secret<tensor<32x32xi16>>)
      attrs = {arg0 = {tensor_ext.layout = #map}, arg1 = {tensor_ext.layout = #map}, layout = [#map1]} {
  ^body(%input0: tensor<32x32xi16>, %input1: tensor<32x32xi16>):
    %1 = tensor_ext.assign_layout %cst {tensor_ext.layout = #map1} : tensor<32xi16>
    %reduced = linalg.reduce { arith.addi }
               ins(%input0 : tensor<32x32xi16>)
               outs(%1 : tensor<32xi16>)
               dimensions = [0] {tensor_ext.layout = [#map1]}

    %2 = tensor_ext.assign_layout %cst_0 {tensor_ext.layout = #map1} : tensor<32xi16>
    %3 = tensor_ext.convert_layout %2 {from_layout = #map1, to_layout = #map2} : tensor<32xi16>
    %reduced_1 = linalg.reduce { arith.addi }
               ins(%input1 : tensor<32x32xi16>)
               outs(%3 : tensor<32xi16>)
               dimensions = [1] {tensor_ext.layout = [#map2]}

    %4 = tensor_ext.convert_layout %reduced_1 {from_layout = #map2, to_layout = #map1} : tensor<32xi16>
    %5 = arith.addi %reduced, %4 {tensor_ext.layout = [#map1]} : tensor<32xi16>
    secret.yield %5 : tensor<32xi16>
  } -> !secret.secret<tensor<32xi16>>
  return %0 : !secret.secret<tensor<32xi16>>
}
```
