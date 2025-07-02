---
title: Secret Capture Generic Ambient Scope
weight: 12
---

## Overview

The `secret-capture-generic-ambient-scope` pass makes implicit dependencies
explicit by capturing values from the ambient scope that are used within
`secret.generic` operations. For each value used in a generic body that is
defined outside the generic, this pass adds it to the argument list of the
generic operation.

## Input/Output

- **Input**: IR with `secret.generic` operations that implicitly capture values
  from surrounding scope
- **Output**: IR with `secret.generic` operations that explicitly pass all
  external dependencies as arguments

## Options

This pass has no configurable options.

## Usage Examples

```bash
heir-opt --secret-capture-generic-ambient-scope input.mlir
```

### Example Input

```mlir
func.func @main(%arg0: !secret.secret<i32>) -> !secret.secret<i32> {
  %c100 = arith.constant 100 : i32
  %0 = secret.generic(%arg0 : !secret.secret<i32>) {
  ^bb0(%arg1: i32):
    %1 = arith.addi %arg1, %c100 : i32  // %c100 is captured from ambient scope
    secret.yield %1 : i32
  } -> !secret.secret<i32>
  return %0 : !secret.secret<i32>
}
```

### Example Output

```mlir
func.func @main(%arg0: !secret.secret<i32>) -> !secret.secret<i32> {
  %c100 = arith.constant 100 : i32
  %0 = secret.generic(%arg0, %c100 : !secret.secret<i32>, i32) {
  ^bb0(%arg1: i32, %arg2: i32):
    %1 = arith.addi %arg1, %arg2 : i32  // %c100 now passed as %arg2
    secret.yield %1 : i32
  } -> !secret.secret<i32>
  return %0 : !secret.secret<i32>
}
```

### More Complex Example

With multiple captured values and nested scopes:

**Input:**

```mlir
func.func @compute(%arg0: !secret.secret<i32>) -> !secret.secret<i32> {
  %c10 = arith.constant 10 : i32
  %c20 = arith.constant 20 : i32
  %0 = secret.generic(%arg0 : !secret.secret<i32>) {
  ^bb0(%arg1: i32):
    %1 = arith.addi %arg1, %c10 : i32
    %2 = arith.muli %1, %c20 : i32
    secret.yield %2 : i32
  } -> !secret.secret<i32>
  return %0 : !secret.secret<i32>
}
```

**Output:**

```mlir
func.func @compute(%arg0: !secret.secret<i32>) -> !secret.secret<i32> {
  %c10 = arith.constant 10 : i32
  %c20 = arith.constant 20 : i32
  %0 = secret.generic(%arg0, %c10, %c20 : !secret.secret<i32>, i32, i32) {
  ^bb0(%arg1: i32, %arg2: i32, %arg3: i32):
    %1 = arith.addi %arg1, %arg2 : i32
    %2 = arith.muli %1, %arg3 : i32
    secret.yield %2 : i32
  } -> !secret.secret<i32>
  return %0 : !secret.secret<i32>
}
```

## When to Use

The `secret-capture-generic-ambient-scope` pass should be used:

1. **Before function extraction** to ensure all dependencies are explicit
1. **To improve IR quality** by eliminating implicit captures that complicate
   analysis
1. **Before passes that require well-formed argument lists** for their
   transformations
1. **To enable function extraction from generic bodies** by making all inputs
   explicit
1. **In preparation for backend lowering** that expects explicit argument
   passing
1. **To simplify dataflow analysis** by making all value dependencies visible

This pass is essential for transformations that need to extract, inline, or
otherwise manipulate the bodies of `secret.generic` operations, as it ensures
all dependencies are explicitly represented in the IR.
