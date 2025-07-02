---
title: Secret Merge Adjacent Generics
weight: 13
---

## Overview

The `secret-merge-adjacent-generics` pass merges two immediately sequential
`secret.generic` operations into a single generic operation. This optimization
reduces the overhead of having multiple separate generic regions and can improve
the efficiency of subsequent transformations.

## Input/Output

- **Input**: IR with consecutive `secret.generic` operations that can be merged
- **Output**: IR with merged `secret.generic` operations combining the
  computation of adjacent generics

## Options

This pass has no configurable options.

## Usage Examples

```bash
heir-opt --secret-merge-adjacent-generics input.mlir
```

### Example Input

```mlir
func.func @main(%arg0: !secret.secret<i32>) -> !secret.secret<i32> {
  %0 = secret.generic(%arg0 : !secret.secret<i32>) {
  ^bb0(%arg1: i32):
    %1 = arith.constant 100 : i32
    %2 = arith.addi %arg1, %1 : i32
    secret.yield %2 : i32
  } -> !secret.secret<i32>

  %3 = secret.generic(%0 : !secret.secret<i32>) {
  ^bb0(%arg2: i32):
    %4 = arith.constant 50 : i32
    %5 = arith.muli %arg2, %4 : i32
    secret.yield %5 : i32
  } -> !secret.secret<i32>

  return %3 : !secret.secret<i32>
}
```

### Example Output

```mlir
func.func @main(%arg0: !secret.secret<i32>) -> !secret.secret<i32> {
  %0 = secret.generic(%arg0 : !secret.secret<i32>) {
  ^bb0(%arg1: i32):
    %1 = arith.constant 100 : i32
    %2 = arith.addi %arg1, %1 : i32
    %3 = arith.constant 50 : i32
    %4 = arith.muli %2, %3 : i32
    secret.yield %4 : i32
  } -> !secret.secret<i32>

  return %0 : !secret.secret<i32>
}
```

### Complex Example with Multiple Operands

**Input:**

```mlir
func.func @compute(%arg0: !secret.secret<i32>, %arg1: !secret.secret<i32>) -> !secret.secret<i32> {
  %0 = secret.generic(%arg0, %arg1 : !secret.secret<i32>, !secret.secret<i32>) {
  ^bb0(%arg2: i32, %arg3: i32):
    %1 = arith.addi %arg2, %arg3 : i32
    secret.yield %1 : i32
  } -> !secret.secret<i32>

  %2 = secret.generic(%0 : !secret.secret<i32>) {
  ^bb0(%arg4: i32):
    %3 = arith.constant 2 : i32
    %4 = arith.muli %arg4, %3 : i32
    secret.yield %4 : i32
  } -> !secret.secret<i32>

  return %2 : !secret.secret<i32>
}
```

**Output:**

```mlir
func.func @compute(%arg0: !secret.secret<i32>, %arg1: !secret.secret<i32>) -> !secret.secret<i32> {
  %0 = secret.generic(%arg0, %arg1 : !secret.secret<i32>, !secret.secret<i32>) {
  ^bb0(%arg2: i32, %arg3: i32):
    %1 = arith.addi %arg2, %arg3 : i32
    %2 = arith.constant 2 : i32
    %3 = arith.muli %1, %2 : i32
    secret.yield %3 : i32
  } -> !secret.secret<i32>

  return %0 : !secret.secret<i32>
}
```

## When to Use

The `secret-merge-adjacent-generics` pass should be used:

1. **After distribution passes** that may have created many small generic
   operations
1. **To reduce overhead** of multiple generic region boundaries
1. **Before backend lowering** to present larger computation units for
   optimization
1. **As a cleanup pass** to simplify IR structure
1. **For testing purposes** to verify the correctness of merge transformations
1. **To improve compilation efficiency** by reducing the number of operations
   that subsequent passes need to process

This pass is particularly useful as a cleanup step in compilation pipelines,
especially after passes like `secret-distribute-generic` that may fragment
computation into many small pieces that can be beneficially recombined.
