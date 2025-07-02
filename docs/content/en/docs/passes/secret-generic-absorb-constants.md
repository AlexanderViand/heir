---
title: Secret Generic Absorb Constants
weight: 14
---

## Overview

The `secret-generic-absorb-constants` pass moves constant value definitions from
the ambient scope into the body of `secret.generic` operations that use them.
This transformation makes the generic bodies more self-contained by
internalizing constant dependencies.

## Input/Output

- **Input**: IR with `secret.generic` operations that use constants defined in
  the surrounding scope
- **Output**: IR with `secret.generic` operations containing internal constant
  definitions

## Options

This pass has no configurable options.

## Usage Examples

```bash
heir-opt --secret-generic-absorb-constants input.mlir
```

### Example Input

```mlir
func.func @main(%arg0: !secret.secret<i32>) -> !secret.secret<i32> {
  %c100 = arith.constant 100 : i32
  %c50 = arith.constant 50 : i32
  %0 = secret.generic(%arg0, %c100, %c50 : !secret.secret<i32>, i32, i32) {
  ^bb0(%arg1: i32, %arg2: i32, %arg3: i32):
    %1 = arith.addi %arg1, %arg2 : i32
    %2 = arith.muli %1, %arg3 : i32
    secret.yield %2 : i32
  } -> !secret.secret<i32>
  return %0 : !secret.secret<i32>
}
```

### Example Output

```mlir
func.func @main(%arg0: !secret.secret<i32>) -> !secret.secret<i32> {
  %0 = secret.generic(%arg0 : !secret.secret<i32>) {
  ^bb0(%arg1: i32):
    %c100 = arith.constant 100 : i32
    %c50 = arith.constant 50 : i32
    %1 = arith.addi %arg1, %c100 : i32
    %2 = arith.muli %1, %c50 : i32
    secret.yield %2 : i32
  } -> !secret.secret<i32>
  return %0 : !secret.secret<i32>
}
```

### Complex Example with Shared Constants

**Input:**

```mlir
func.func @compute(%arg0: !secret.secret<i32>, %arg1: !secret.secret<i32>) -> (!secret.secret<i32>, !secret.secret<i32>) {
  %c10 = arith.constant 10 : i32
  %c20 = arith.constant 20 : i32

  %0 = secret.generic(%arg0, %c10 : !secret.secret<i32>, i32) {
  ^bb0(%arg2: i32, %arg3: i32):
    %1 = arith.addi %arg2, %arg3 : i32
    secret.yield %1 : i32
  } -> !secret.secret<i32>

  %2 = secret.generic(%arg1, %c20 : !secret.secret<i32>, i32) {
  ^bb0(%arg4: i32, %arg5: i32):
    %3 = arith.muli %arg4, %arg5 : i32
    secret.yield %3 : i32
  } -> !secret.secret<i32>

  return %0, %2 : !secret.secret<i32>, !secret.secret<i32>
}
```

**Output:**

```mlir
func.func @compute(%arg0: !secret.secret<i32>, %arg1: !secret.secret<i32>) -> (!secret.secret<i32>, !secret.secret<i32>) {
  %0 = secret.generic(%arg0 : !secret.secret<i32>) {
  ^bb0(%arg2: i32):
    %c10 = arith.constant 10 : i32
    %1 = arith.addi %arg2, %c10 : i32
    secret.yield %1 : i32
  } -> !secret.secret<i32>

  %2 = secret.generic(%arg1 : !secret.secret<i32>) {
  ^bb0(%arg4: i32):
    %c20 = arith.constant 20 : i32
    %3 = arith.muli %arg4, %c20 : i32
    secret.yield %3 : i32
  } -> !secret.secret<i32>

  return %0, %2 : !secret.secret<i32>, !secret.secret<i32>
}
```

## When to Use

The `secret-generic-absorb-constants` pass should be used:

1. **Before function extraction** to ensure extracted functions contain all
   necessary constants
1. **After `secret-capture-generic-ambient-scope`** to internalize constants
   that were captured as arguments
1. **To improve locality** by keeping constants close to their usage
1. **Before backend lowering** that expects self-contained computation regions
1. **To simplify dataflow analysis** by reducing external dependencies
1. **In preparation for optimization passes** that work better with local
   constants

This pass works particularly well in combination with
`secret-extract-generic-body`, as noted in that pass's documentation, ensuring
that extracted functions contain any constants they need rather than depending
on external definitions.
