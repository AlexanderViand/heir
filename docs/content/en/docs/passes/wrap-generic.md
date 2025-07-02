---
title: Wrap Generic
weight: 2
---

## Overview

The `wrap-generic` pass converts functions with `{secret.secret}` annotated
arguments to use `!secret.secret<...>` types and wraps the function body in a
`secret.generic` region. This transforms plaintext functions with secret
annotations into the Secret dialect's structured representation for encrypted
computation.

## Input/Output

- **Input**: Functions with `{secret.secret}` argument attributes and standard
  types
- **Output**: Functions using `!secret.secret<...>` types with bodies wrapped in
  `secret.generic` operations

## Options

This pass has no configurable options.

## Usage Examples

```bash
heir-opt --wrap-generic input.mlir
```

### Example Input

```mlir
func.func @main(%arg0: i32 {secret.secret}) -> i32 {
  %0 = arith.constant 100 : i32
  %1 = arith.addi %0, %arg0 : i32
  return %1 : i32
}
```

### Example Output

```mlir
func.func @main(%arg0: !secret.secret<i32>) -> !secret.secret<i32> {
  %0 = secret.generic(%arg0 : !secret.secret<i32>) {
  ^bb0(%arg1: i32):
    %1 = arith.constant 100 : i32
    %2 = arith.addi %1, %arg1 : i32
    secret.yield %2 : i32
  } -> !secret.secret<i32>
  return %0 : !secret.secret<i32>
}
```

### More Complex Example

For functions with multiple secret arguments and mixed computation:

**Input:**

```mlir
func.func @compute(%x: tensor<8xi16> {secret.secret}, %y: i32 {secret.secret}) -> tensor<8xi16> {
  %c1 = arith.constant 1 : i32
  %sum = arith.addi %y, %c1 : i32
  %broadcasted = tensor.splat %sum : tensor<8xi16>
  %result = arith.addi %x, %broadcasted : tensor<8xi16>
  return %result : tensor<8xi16>
}
```

**Output:**

```mlir
func.func @compute(%x: !secret.secret<tensor<8xi16>>, %y: !secret.secret<i32>) -> !secret.secret<tensor<8xi16>> {
  %0 = secret.generic(%x, %y : !secret.secret<tensor<8xi16>>, !secret.secret<i32>) {
  ^bb0(%arg0: tensor<8xi16>, %arg1: i32):
    %c1 = arith.constant 1 : i32
    %sum = arith.addi %arg1, %c1 : i32
    %broadcasted = tensor.splat %sum : tensor<8xi16>
    %result = arith.addi %arg0, %broadcasted : tensor<8xi16>
    secret.yield %result : tensor<8xi16>
  } -> !secret.secret<tensor<8xi16>>
  return %0 : !secret.secret<tensor<8xi16>>
}
```

## When to Use

The `wrap-generic` pass should be used:

1. **After the `secretize` pass** that adds `{secret.secret}` annotations to
   function arguments
1. **As the second step in FHE compilation pipelines** to establish the Secret
   dialect structure
1. **Before secret distribution and optimization passes** that operate on
   `secret.generic` regions
1. **When transitioning from plaintext IR** with secret annotations to the full
   Secret dialect representation
1. **Before applying secret-specific transformations** like
   `secret-distribute-generic` or `secret-capture-generic-ambient-scope`

This pass creates the fundamental Secret dialect structure that subsequent
passes can analyze and optimize for different FHE schemes.
