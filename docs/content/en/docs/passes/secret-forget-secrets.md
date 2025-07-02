---
title: Secret Forget Secrets
weight: 10
---

## Overview

The `secret-forget-secrets` pass converts secret types back to standard types by
removing the `!secret.secret<...>` type wrapper and converting encrypted
computation back to cleartext computation. This pass effectively reverses
secretization, making it useful for testing, validation, and creating reference
implementations.

## Input/Output

- **Input**: IR with `!secret.secret<T>` types and `secret.generic` operations
- **Output**: IR with standard types `T` and cleartext computation operations

## Options

This pass has no configurable options.

## Usage Examples

```bash
heir-opt --secret-forget-secrets input.mlir
```

### Example Input

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

### Example Output

```mlir
func.func @main(%arg0: i32) -> i32 {
  %1 = arith.constant 100 : i32
  %2 = arith.addi %1, %arg0 : i32
  return %2 : i32
}
```

## When to Use

The `secret-forget-secrets` pass should be used:

1. **For testing and validation** to generate cleartext reference
   implementations from secret computations
1. **For debugging** to isolate issues in secret dialect logic vs. underlying
   computation
1. **For benchmarking** to compare cleartext vs. encrypted performance
   characteristics
1. **During development** to verify the correctness of secret transformations
1. **For creating test oracles** where you need both encrypted and cleartext
   versions of the same computation

This pass is particularly valuable in testing pipelines where you want to verify
that your encrypted computation produces the same results as the equivalent
cleartext computation.
