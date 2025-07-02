---
title: Secretize
weight: 1
---

## Overview

The `secretize` pass adds secret argument attributes to entry function
arguments, marking them as secret values that require encryption. This is
typically the first step in a compilation pipeline for homomorphic encryption,
identifying which function inputs should be treated as encrypted data.

## Input/Output

- **Input**: MLIR module with standard function arguments
- **Output**: MLIR module with function arguments annotated with
  `{secret.secret}` attributes

## Options

- `--function=<function_name>`: Apply secretization to a specific function only.
  If not specified, applies to all functions in the module.

## Usage Examples

Apply to all functions in a module:

```bash
heir-opt --secretize input.mlir
```

Apply to a specific function:

```bash
heir-opt --secretize="function=main" input.mlir
```

### Example Input

```mlir
func.func @main(%arg0: i32, %arg1: tensor<8xi16>) -> i32 {
  %0 = arith.addi %arg0, %arg0 : i32
  return %0 : i32
}
```

### Example Output

```mlir
func.func @main(%arg0: i32 {secret.secret}, %arg1: tensor<8xi16> {secret.secret}) -> i32 {
  %0 = arith.addi %arg0, %arg0 : i32
  return %0 : i32
}
```

## When to Use

The `secretize` pass should be used:

1. **At the beginning of FHE compilation pipelines** to mark which function
   arguments represent encrypted inputs
1. **Before other secret dialect transformations** like `wrap-generic` that
   depend on secret annotations
1. **When you want to automatically annotate all function arguments** as secret
   without manually adding attributes
1. **For prototyping and testing** when you want to quickly convert a plaintext
   function to work with the secret dialect

This pass is typically followed by `wrap-generic` to convert the annotated
functions to use secret types and wrap their bodies in `secret.generic`
operations.
