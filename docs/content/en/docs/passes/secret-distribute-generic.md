---
title: Secret Distribute Generic
weight: 11
---

## Overview

The `secret-distribute-generic` pass breaks down large `secret.generic`
operations into smaller, more manageable sequences of generic operations, with
each containing a single operation. This transformation enables fine-grained
analysis and optimization of individual operations within encrypted computation
regions.

## Input/Output

- **Input**: IR with large `secret.generic` operations containing multiple
  operations
- **Output**: IR with sequences of smaller `secret.generic` operations, each
  containing a single operation

## Options

- `--distribute-through=<op_list>`: Comma-separated list of operation names
  (e.g., `"affine.for,scf.if"`) that the pass should distribute through. If
  unset, distributes through all operations when possible.

## Usage Examples

Distribute through all operations:

```bash
heir-opt --secret-distribute-generic input.mlir
```

Distribute through specific operations only:

```bash
heir-opt --secret-distribute-generic="distribute-through=affine.for,scf.if" input.mlir
```

### Example Input

```mlir
func.func @main(%arg0: !secret.secret<i32>, %arg1: !secret.secret<i32>) -> !secret.secret<i32> {
  %0 = secret.generic(%arg0, %arg1 : !secret.secret<i32>, !secret.secret<i32>) {
  ^bb0(%arg2: i32, %arg3: i32):
    %1 = arith.constant 100 : i32
    %2 = arith.addi %arg2, %1 : i32
    %3 = arith.muli %2, %arg3 : i32
    secret.yield %3 : i32
  } -> !secret.secret<i32>
  return %0 : !secret.secret<i32>
}
```

### Example Output

```mlir
func.func @main(%arg0: !secret.secret<i32>, %arg1: !secret.secret<i32>) -> !secret.secret<i32> {
  %0 = arith.constant 100 : i32
  %1 = secret.generic(%arg0 : !secret.secret<i32>) {
  ^bb0(%arg2: i32):
    %2 = arith.addi %arg2, %0 : i32
    secret.yield %2 : i32
  } -> !secret.secret<i32>
  %3 = secret.generic(%1, %arg1 : !secret.secret<i32>, !secret.secret<i32>) {
  ^bb0(%arg3: i32, %arg4: i32):
    %4 = arith.muli %arg3, %arg4 : i32
    secret.yield %4 : i32
  } -> !secret.secret<i32>
  return %3 : !secret.secret<i32>
}
```

## When to Use

The `secret-distribute-generic` pass should be used:

1. **After initial secretization** when you have large generic regions that need
   to be broken down
1. **Before backend-specific transformations** that work better on individual
   operations
1. **To enable fine-grained analysis** of individual secret operations
1. **For optimization passes** that need to reason about specific operation
   patterns
1. **To prepare for parallelization** by exposing independent operations
1. **In front-end pipelines** to simplify complex secret regions into manageable
   units

This pass is particularly useful in compilation pipelines where subsequent
transformations need to analyze or modify individual operations rather than
entire computation regions.
