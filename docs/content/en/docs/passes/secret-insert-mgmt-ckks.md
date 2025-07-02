---
title: Secret Insert Mgmt CKKS
weight: 21
---

## Overview

The `secret-insert-mgmt-ckks` pass inserts CKKS ciphertext management operations
including relinearization, rescaling (modulus reduction), and bootstrapping. It
implements a management strategy similar to BGV but uses `mgmt.modreduce` to
represent CKKS rescaling operations and includes bootstrap insertion when levels
are exhausted.

## Input/Output

- **Input**: IR with `secret.generic` operations containing arithmetic
  operations on secret values
- **Output**: IR with `mgmt.relinearize`, `mgmt.modreduce` (rescaling), and
  potentially bootstrap operations inserted, plus `mgmt.mgmt` attributes

## Options

- `--after-mul`: Insert rescaling after each multiplication (default: false)
- `--before-mul-include-first-mul`: Insert rescaling before each multiplication,
  including the first one (default: false)
- `--slot-number=<int>`: Default number of slots for ciphertext space (default:
  1024\)
- `--bootstrap-waterline=<int>`: Level threshold for inserting bootstrap
  operations (default: 10)

## Usage Examples

Basic usage with default settings:

```bash
heir-opt --secret-insert-mgmt-ckks input.mlir
```

With custom slot number and bootstrap waterline:

```bash
heir-opt --secret-insert-mgmt-ckks="slot-number=2048,bootstrap-waterline=5" input.mlir
```

With rescaling after multiplications:

```bash
heir-opt --secret-insert-mgmt-ckks="after-mul=true" input.mlir
```

### Example Input

```mlir
func.func @main(%arg0: !secret.secret<tensor<8xf64>>, %arg1: !secret.secret<tensor<8xf64>>) -> !secret.secret<tensor<8xf64>> {
  %0 = secret.generic(%arg0, %arg1 : !secret.secret<tensor<8xf64>>, !secret.secret<tensor<8xf64>>) {
  ^bb0(%arg2: tensor<8xf64>, %arg3: tensor<8xf64>):
    %1 = arith.mulf %arg2, %arg3 : tensor<8xf64>
    %c2 = arith.constant dense<2.0> : tensor<8xf64>
    %2 = arith.addf %1, %c2 : tensor<8xf64>
    secret.yield %2 : tensor<8xf64>
  } -> !secret.secret<tensor<8xf64>>
  return %0 : !secret.secret<tensor<8xf64>>
}
```

### Example Output

```mlir
func.func @main(%arg0: !secret.secret<tensor<8xf64>>, %arg1: !secret.secret<tensor<8xf64>>) -> !secret.secret<tensor<8xf64>> {
  %0 = secret.generic(%arg0, %arg1 : !secret.secret<tensor<8xf64>>, !secret.secret<tensor<8xf64>>)
      attrs = {arg0 = {mgmt.mgmt = #mgmt.mgmt<level = 1>},
               arg1 = {mgmt.mgmt = #mgmt.mgmt<level = 1>}} {
  ^bb0(%arg2: tensor<8xf64>, %arg3: tensor<8xf64>):
    // Multiplication with relinearization and rescaling
    %1 = arith.mulf %arg2, %arg3 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : tensor<8xf64>
    %2 = mgmt.relinearize %1 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xf64>

    %c2 = arith.constant dense<2.0> : tensor<8xf64>
    %3 = arith.addf %2, %c2 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xf64>

    // Rescaling before yielding result of multiplication-derived value
    %4 = mgmt.modreduce %3 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<8xf64>
    secret.yield %4 : tensor<8xf64>
  } -> !secret.secret<tensor<8xf64>>
  return %0 : !secret.secret<tensor<8xf64>>
}
```

### Complex Example with Polynomial Evaluation

**Input:**

```mlir
func.func @polynomial_ckks(%arg0: !secret.secret<tensor<4xf64>>) -> !secret.secret<tensor<4xf64>> {
  %0 = secret.generic(%arg0 : !secret.secret<tensor<4xf64>>) {
  ^bb0(%arg1: tensor<4xf64>):
    // Evaluate polynomial: 3x^3 + 2x^2 + x + 1
    %c1 = arith.constant dense<1.0> : tensor<4xf64>
    %c2 = arith.constant dense<2.0> : tensor<4xf64>
    %c3 = arith.constant dense<3.0> : tensor<4xf64>

    %x_squared = arith.mulf %arg1, %arg1 : tensor<4xf64>
    %x_cubed = arith.mulf %x_squared, %arg1 : tensor<4xf64>

    %term1 = arith.mulf %x_cubed, %c3 : tensor<4xf64>
    %term2 = arith.mulf %x_squared, %c2 : tensor<4xf64>

    %sum1 = arith.addf %term1, %term2 : tensor<4xf64>
    %sum2 = arith.addf %sum1, %arg1 : tensor<4xf64>
    %result = arith.addf %sum2, %c1 : tensor<4xf64>

    secret.yield %result : tensor<4xf64>
  } -> !secret.secret<tensor<4xf64>>
  return %0 : !secret.secret<tensor<4xf64>>
}
```

**Output:**

```mlir
func.func @polynomial_ckks(%arg0: !secret.secret<tensor<4xf64>>) -> !secret.secret<tensor<4xf64>> {
  %0 = secret.generic(%arg0 : !secret.secret<tensor<4xf64>>)
      attrs = {arg0 = {mgmt.mgmt = #mgmt.mgmt<level = 3>}} {
  ^bb0(%arg1: tensor<4xf64>):
    %c1 = arith.constant dense<1.0> : tensor<4xf64>
    %c2 = arith.constant dense<2.0> : tensor<4xf64>
    %c3 = arith.constant dense<3.0> : tensor<4xf64>

    // x^2 computation
    %x_squared = arith.mulf %arg1, %arg1 {mgmt.mgmt = #mgmt.mgmt<level = 3, dimension = 3>} : tensor<4xf64>
    %1 = mgmt.relinearize %x_squared {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<4xf64>

    // x^3 computation
    %x_cubed = arith.mulf %1, %arg1 {mgmt.mgmt = #mgmt.mgmt<level = 3, dimension = 3>} : tensor<4xf64>
    %2 = mgmt.relinearize %x_cubed {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<4xf64>

    // Coefficient multiplications
    %term1 = arith.mulf %2, %c3 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<4xf64>
    %term2 = arith.mulf %1, %c2 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<4xf64>

    // Rescaling for level matching
    %3 = mgmt.modreduce %term1 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<4xf64>
    %4 = mgmt.modreduce %term2 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<4xf64>
    %5 = mgmt.modreduce %arg1 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<4xf64>

    // Additions
    %sum1 = arith.addf %3, %4 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<4xf64>
    %sum2 = arith.addf %sum1, %5 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<4xf64>
    %result = arith.addf %sum2, %c1 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<4xf64>

    %6 = mgmt.modreduce %result {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<4xf64>
    secret.yield %6 : tensor<4xf64>
  } -> !secret.secret<tensor<4xf64>>
  return %0 : !secret.secret<tensor<4xf64>>
}
```

### Bootstrap Insertion Example

When levels are exhausted (reach the bootstrap waterline), bootstrap operations
are automatically inserted:

**Input with deep computation:**

```mlir
func.func @deep_computation(%arg0: !secret.secret<tensor<2xf64>>) -> !secret.secret<tensor<2xf64>> {
  %0 = secret.generic(%arg0 : !secret.secret<tensor<2xf64>>) {
  ^bb0(%arg1: tensor<2xf64>):
    // Deep polynomial that exhausts levels
    %c2 = arith.constant dense<2.0> : tensor<2xf64>

    // Many consecutive multiplications
    %1 = arith.mulf %arg1, %c2 : tensor<2xf64>
    %2 = arith.mulf %1, %c2 : tensor<2xf64>
    %3 = arith.mulf %2, %c2 : tensor<2xf64>
    // ... (many more operations leading to bootstrap)

    secret.yield %3 : tensor<2xf64>
  } -> !secret.secret<tensor<2xf64>>
  return %0 : !secret.secret<tensor<2xf64>>
}
```

**Output (with bootstrap-waterline=1):**

```mlir
func.func @deep_computation(%arg0: !secret.secret<tensor<2xf64>>) -> !secret.secret<tensor<2xf64>> {
  %0 = secret.generic(%arg0 : !secret.secret<tensor<2xf64>>)
      attrs = {arg0 = {mgmt.mgmt = #mgmt.mgmt<level = 3>}} {
  ^bb0(%arg1: tensor<2xf64>):
    %c2 = arith.constant dense<2.0> : tensor<2xf64>

    %1 = arith.mulf %arg1, %c2 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : tensor<2xf64>
    %2 = mgmt.modreduce %1 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<2xf64>

    %3 = arith.mulf %2, %c2 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<2xf64>
    %4 = mgmt.modreduce %3 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<2xf64>

    // Bootstrap inserted when level reaches waterline
    %5 = ckks.bootstrap %4 {mgmt.mgmt = #mgmt.mgmt<level = 10>} : tensor<2xf64>

    %6 = arith.mulf %5, %c2 {mgmt.mgmt = #mgmt.mgmt<level = 10>} : tensor<2xf64>
    %7 = mgmt.modreduce %6 {mgmt.mgmt = #mgmt.mgmt<level = 9>} : tensor<2xf64>

    secret.yield %7 : tensor<2xf64>
  } -> !secret.secret<tensor<2xf64>>
  return %0 : !secret.secret<tensor<2xf64>>
}
```

## CKKS Management Strategy

The pass implements:

1. **Relinearization**: After ciphertext-ciphertext multiplications to maintain
   dimension 2
1. **Rescaling**: Using `mgmt.modreduce` to represent CKKS rescaling operations
1. **Level Matching**: Ensuring operands have the same level for additions
1. **Bootstrap Insertion**: Automatic insertion when levels are exhausted
   (greedy policy)
1. **Slot Management**: Tracking the number of SIMD slots in CKKS ciphertexts

## When to Use

The `secret-insert-mgmt-ckks` pass should be used:

1. **Before lowering to CKKS backend** to insert necessary ciphertext management
   operations
1. **For CKKS scheme implementations** that require explicit rescaling and
   relinearization
1. **When working with approximate arithmetic** where rescaling is crucial for
   noise management
1. **To handle deep computations** that may require bootstrapping
1. **After secretization and generic distribution** when you have well-formed
   secret operations
1. **In CKKS compilation pipelines** as preparation for backend-specific
   lowering

This pass is essential for CKKS homomorphic encryption, managing the complex
interplay between rescaling, relinearization, and bootstrapping required for
successful CKKS computations.
