---
title: Secret Extract Generic Body
weight: 16
---

## Overview

The `secret-extract-generic-body` pass extracts the body of all `secret.generic`
operations into separate functions and replaces the generic bodies with function
calls. This transformation is useful for modularity, code reuse, and simplifying
complex generic operations by breaking them into named, reusable functions.

## Input/Output

- **Input**: IR with `secret.generic` operations containing computation in their
  bodies
- **Output**: IR with `secret.generic` operations containing only function
  calls, and new function definitions for the extracted bodies

## Options

This pass has no configurable options.

## Usage Examples

```bash
heir-opt --secret-extract-generic-body input.mlir
```

This pass works best when `--secret-generic-absorb-constants` is run before it
to ensure that extracted functions contain any constants they need.

### Example Input

```mlir
func.func @main(%arg0: !secret.secret<i32>) -> !secret.secret<i32> {
  %0 = secret.generic(%arg0 : !secret.secret<i32>) {
  ^bb0(%arg1: i32):
    %c100 = arith.constant 100 : i32
    %1 = arith.addi %arg1, %c100 : i32
    %2 = arith.muli %1, %arg1 : i32
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
    %1 = func.call @__secret_generic_0(%arg1) : (i32) -> i32
    secret.yield %1 : i32
  } -> !secret.secret<i32>
  return %0 : !secret.secret<i32>
}

func.func private @__secret_generic_0(%arg0: i32) -> i32 {
  %c100 = arith.constant 100 : i32
  %0 = arith.addi %arg0, %c100 : i32
  %1 = arith.muli %0, %arg0 : i32
  return %1 : i32
}
```

### Complex Example with Multiple Generics

**Input:**

```mlir
func.func @compute(%arg0: !secret.secret<i32>, %arg1: !secret.secret<i32>) -> !secret.secret<i32> {
  %0 = secret.generic(%arg0 : !secret.secret<i32>) {
  ^bb0(%arg2: i32):
    %c10 = arith.constant 10 : i32
    %1 = arith.addi %arg2, %c10 : i32
    secret.yield %1 : i32
  } -> !secret.secret<i32>

  %2 = secret.generic(%0, %arg1 : !secret.secret<i32>, !secret.secret<i32>) {
  ^bb0(%arg3: i32, %arg4: i32):
    %3 = arith.muli %arg3, %arg4 : i32
    %c2 = arith.constant 2 : i32
    %4 = arith.divi %3, %c2 : i32
    secret.yield %4 : i32
  } -> !secret.secret<i32>

  return %2 : !secret.secret<i32>
}
```

**Output:**

```mlir
func.func @compute(%arg0: !secret.secret<i32>, %arg1: !secret.secret<i32>) -> !secret.secret<i32> {
  %0 = secret.generic(%arg0 : !secret.secret<i32>) {
  ^bb0(%arg2: i32):
    %1 = func.call @__secret_generic_0(%arg2) : (i32) -> i32
    secret.yield %1 : i32
  } -> !secret.secret<i32>

  %2 = secret.generic(%0, %arg1 : !secret.secret<i32>, !secret.secret<i32>) {
  ^bb0(%arg3: i32, %arg4: i32):
    %3 = func.call @__secret_generic_1(%arg3, %arg4) : (i32, i32) -> i32
    secret.yield %3 : i32
  } -> !secret.secret<i32>

  return %2 : !secret.secret<i32>
}

func.func private @__secret_generic_0(%arg0: i32) -> i32 {
  %c10 = arith.constant 10 : i32
  %0 = arith.addi %arg0, %c10 : i32
  return %0 : i32
}

func.func private @__secret_generic_1(%arg0: i32, %arg1: i32) -> i32 {
  %0 = arith.muli %arg0, %arg1 : i32
  %c2 = arith.constant 2 : i32
  %1 = arith.divi %0, %c2 : i32
  return %1 : i32
}
```

## When to Use

The `secret-extract-generic-body` pass should be used:

1. **For modular code organization** by breaking complex generic bodies into
   named functions
1. **To enable code reuse** when similar computation patterns appear in multiple
   generics
1. **For testing and debugging** by making individual computation units testable
   in isolation
1. **Before backend lowering** that works better with function-based
   organization
1. **To simplify analysis** by converting complex inline computation to function
   calls
1. **After other generic preparation passes** like
   `secret-generic-absorb-constants` and `secret-capture-generic-ambient-scope`

This pass is particularly useful in compilation pipelines where you want to
modularize the computation within secret regions, making the IR more
maintainable and potentially enabling optimizations at the function level.
