---
title: Convert Secret For to Static For
weight: 6
---

## Overview

The `convert-secret-for-to-static-for` pass converts `scf.for` loops with
secret-dependent bounds into `affine.for` loops with constant bounds. This
transformation is essential for creating data-oblivious programs that execute in
constant time, regardless of secret input values.

## Input/Output

- **Input**: MLIR IR containing `scf.for` operations with secret-dependent
  bounds, annotated with `{lower = X, upper = Y}` attributes
- **Output**: MLIR IR with `affine.for` operations using constant bounds and
  conditional execution via `scf.if` operations

## Options

This pass takes no command-line options.

## Usage Examples

```bash
# Convert secret-dependent for loops to static bounds
heir-opt --convert-secret-for-to-static-for input.mlir

# Common usage in data-oblivious pipeline
heir-opt --convert-secret-for-to-static-for --convert-if-to-select input.mlir
```

## When to Use

This pass is essential when:

1. **Creating data-oblivious programs**: When loop bounds depend on secret
   values but constant-time execution is required
1. **FHE circuit generation**: Before lowering to FHE schemes that require
   statically-known loop bounds
1. **Security-critical applications**: When execution time must not leak
   information about secret inputs
1. **Preprocessing for convert-if-to-select**: This pass creates conditional
   operations that need further transformation

The pass must be used in conjunction with `convert-if-to-select` to achieve
complete data-oblivious execution, as it introduces conditional operations that
still depend on secret values.

## Implementation Details

The transformation works by:

1. **Bound Analysis**: Uses `{lower, upper}` annotations on `scf.for` operations
   to determine constant bounds
1. **Loop Restructuring**: Replaces variable bounds with constant bounds
   spanning the maximum possible range
1. **Conditional Execution**: Inserts bounds checks and conditional execution to
   preserve original semantics
1. **Value Propagation**: Maintains proper data flow through conditional yields

The pass ensures that:

- All loop iterations execute (constant-time property)
- Original loop semantics are preserved through conditional execution
- Values are properly propagated through `iter_args` and conditional yields
