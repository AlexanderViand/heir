---
title: Secret Insert Mgmt BGV
weight: 19
---

## Overview

The `secret-insert-mgmt-bgv` pass inserts BGV ciphertext management operations
including relinearization and modulus switching. It implements a placement
strategy that ensures ciphertexts remain linear after multiplications and
manages levels appropriately for BGV homomorphic encryption scheme operations.

## Input/Output

- **Input**: IR with `secret.generic` operations containing arithmetic
  operations on secret values
- **Output**: IR with `mgmt.relinearize` and `mgmt.modreduce` operations
  inserted, plus `mgmt.mgmt` attributes annotating level and dimension
  information

## Options

- `--after-mul`: Insert modulus switching after each multiplication (default:
  false)
- `--before-mul-include-first-mul`: Insert modulus switching before each
  multiplication, including the first one (default: false)

## Usage Examples

Basic usage with default settings:

```bash
heir-opt --secret-insert-mgmt-bgv input.mlir
```

With modulus switching after multiplications:

```bash
heir-opt --secret-insert-mgmt-bgv="after-mul=true" input.mlir
```

With modulus switching before multiplications:

```bash
heir-opt --secret-insert-mgmt-bgv="before-mul-include-first-mul=true" input.mlir
```

### Example Input

```mlir
func.func @main(%arg0: !secret.secret<i16>, %arg1: !secret.secret<i16>) -> !secret.secret<i16> {
  %0 = secret.generic(%arg0, %arg1 : !secret.secret<i16>, !secret.secret<i16>) {
  ^bb0(%arg2: i16, %arg3: i16):
    %1 = arith.muli %arg2, %arg3 : i16
    %2 = arith.addi %1, %arg3 : i16
    secret.yield %2 : i16
  } -> !secret.secret<i16>
  return %0 : !secret.secret<i16>
}
```

### Example Output

```mlir
func.func @main(%arg0: !secret.secret<i16>, %arg1: !secret.secret<i16>) -> !secret.secret<i16> {
  %0 = secret.generic(%arg0, %arg1 : !secret.secret<i16>, !secret.secret<i16>)
      attrs = {arg0 = {mgmt.mgmt = #mgmt.mgmt<level = 1>},
               arg1 = {mgmt.mgmt = #mgmt.mgmt<level = 1>}} {
  ^bb0(%arg2: i16, %arg3: i16):
    // Multiplication increases dimension to 3, requiring relinearization
    %1 = arith.muli %arg2, %arg3 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : i16
    // Relinearization reduces dimension back to 2
    %2 = mgmt.relinearize %1 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : i16
    // Addition with level matching
    %3 = arith.addi %2, %arg3 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : i16
    // Modulus reduction before yielding result of multiplication-derived value
    %4 = mgmt.modreduce %3 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : i16
    secret.yield %4 : i16
  } -> !secret.secret<i16>
  return %0 : !secret.secret<i16>
}
```

### Complex Example with Multiple Operations

**Input:**

```mlir
func.func @polynomial(%arg0: !secret.secret<i32>) -> !secret.secret<i32> {
  %0 = secret.generic(%arg0 : !secret.secret<i32>) {
  ^bb0(%arg1: i32):
    // Compute x^2 + 3x + 2
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32

    %x_squared = arith.muli %arg1, %arg1 : i32
    %three_x = arith.muli %arg1, %c3 : i32
    %sum1 = arith.addi %x_squared, %three_x : i32
    %result = arith.addi %sum1, %c2 : i32

    secret.yield %result : i32
  } -> !secret.secret<i32>
  return %0 : !secret.secret<i32>
}
```

**Output:**

```mlir
func.func @polynomial(%arg0: !secret.secret<i32>) -> !secret.secret<i32> {
  %0 = secret.generic(%arg0 : !secret.secret<i32>)
      attrs = {arg0 = {mgmt.mgmt = #mgmt.mgmt<level = 2>}} {
  ^bb0(%arg1: i32):
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32

    // Ciphertext-ciphertext multiplication
    %x_squared = arith.muli %arg1, %arg1 {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>} : i32
    %1 = mgmt.relinearize %x_squared {mgmt.mgmt = #mgmt.mgmt<level = 2>} : i32

    // Ciphertext-plaintext multiplication
    %three_x = arith.muli %arg1, %c3 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : i32

    // Addition requires level matching
    %2 = mgmt.modreduce %1 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : i32
    %3 = mgmt.modreduce %three_x {mgmt.mgmt = #mgmt.mgmt<level = 1>} : i32
    %sum1 = arith.addi %2, %3 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : i32

    %result = arith.addi %sum1, %c2 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : i32
    %4 = mgmt.modreduce %result {mgmt.mgmt = #mgmt.mgmt<level = 0>} : i32

    secret.yield %4 : i32
  } -> !secret.secret<i32>
  return %0 : !secret.secret<i32>
}
```

## BGV Management Strategy

The pass implements the following strategy:

1. **Relinearization**: After every ciphertext-ciphertext multiplication, a
   `mgmt.relinearize` operation is inserted to keep ciphertexts linear
   (dimension 2).

1. **Modulus Switching**:

   - Inserted before multiplications (controlled by options)
   - Used for level matching in binary operations
   - Applied before yielding results derived from multiplications

1. **Level Management**: Tracks and manages the level (number of remaining
   modulus switches) throughout the computation.

1. **Dimension Tracking**: Monitors ciphertext dimension, with multiplications
   increasing dimension to 3 and relinearization reducing it back to 2.

## When to Use

The `secret-insert-mgmt-bgv` pass should be used:

1. **Before lowering to BGV backend** to insert necessary ciphertext management
   operations
1. **After secretization and generic distribution** when you have well-formed
   secret operations
1. **For BGV scheme implementations** that require explicit management
   operations
1. **To prepare for parameter generation** as the level information guides
   parameter selection
1. **In BGV compilation pipelines** as a crucial step before backend-specific
   lowering

This pass is essential for BGV homomorphic encryption, ensuring that ciphertexts
remain manageable and that the necessary cryptographic operations are properly
inserted into the computation.
