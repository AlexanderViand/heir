---
title: Secret Insert Mgmt BFV
weight: 20
---

## Overview

The `secret-insert-mgmt-bfv` pass inserts BFV ciphertext management operations,
specifically relinearization after multiplications, and computes multiplicative
depth information. While BFV is not typically a leveled scheme like BGV,
tracking multiplicative depth is important for parameter selection and noise
analysis.

## Input/Output

- **Input**: IR with `secret.generic` operations containing arithmetic
  operations on secret values
- **Output**: IR with `mgmt.relinearize` operations inserted after
  multiplications, plus `mgmt.mgmt` attributes with level (multiplicative depth)
  information

## Options

This pass has no configurable options.

## Usage Examples

```bash
heir-opt --secret-insert-mgmt-bfv input.mlir
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
  ^bb0(%input0: i16, %input1: i16):
    // Multiplication increases dimension, requires relinearization
    %1 = arith.muli %input0, %input1 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : i16
    // Relinearization reduces dimension back to 2
    %2 = mgmt.relinearize %1 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : i16
    // Addition preserves level
    %3 = arith.addi %2, %input1 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : i16
    secret.yield %3 : i16
  } -> !secret.secret<i16>
  return %0 : !secret.secret<i16>
}
```

### Complex Example with Nested Multiplications

**Input:**

```mlir
func.func @deep_polynomial(%arg0: !secret.secret<i32>) -> !secret.secret<i32> {
  %0 = secret.generic(%arg0 : !secret.secret<i32>) {
  ^bb0(%arg1: i32):
    // Compute x^4 = (x^2)^2
    %c5 = arith.constant 5 : i32

    %x_squared = arith.muli %arg1, %arg1 : i32
    %x_fourth = arith.muli %x_squared, %x_squared : i32
    %result = arith.addi %x_fourth, %c5 : i32

    secret.yield %result : i32
  } -> !secret.secret<i32>
  return %0 : !secret.secret<i32>
}
```

**Output:**

```mlir
func.func @deep_polynomial(%arg0: !secret.secret<i32>) -> !secret.secret<i32> {
  %0 = secret.generic(%arg0 : !secret.secret<i32>)
      attrs = {arg0 = {mgmt.mgmt = #mgmt.mgmt<level = 2>}} {
  ^bb0(%arg1: i32):
    %c5 = arith.constant 5 : i32

    // First multiplication: level 2, dimension 3
    %x_squared = arith.muli %arg1, %arg1 {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>} : i32
    %1 = mgmt.relinearize %x_squared {mgmt.mgmt = #mgmt.mgmt<level = 2>} : i32

    // Second multiplication: level 2, dimension 3
    %x_fourth = arith.muli %1, %1 {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>} : i32
    %2 = mgmt.relinearize %x_fourth {mgmt.mgmt = #mgmt.mgmt<level = 2>} : i32

    // Addition with constant
    %result = arith.addi %2, %c5 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : i32

    secret.yield %result : i32
  } -> !secret.secret<i32>
  return %0 : !secret.secret<i32>
}
```

### Multiplicative Depth Tracking Example

**Input:**

```mlir
func.func @depth_example(%a: !secret.secret<i32>, %b: !secret.secret<i32>, %c: !secret.secret<i32>) -> !secret.secret<i32> {
  %0 = secret.generic(%a, %b, %c : !secret.secret<i32>, !secret.secret<i32>, !secret.secret<i32>) {
  ^bb0(%arg0: i32, %arg1: i32, %arg2: i32):
    // Path 1: a * b (depth 1)
    %ab = arith.muli %arg0, %arg1 : i32

    // Path 2: c * c (depth 1)
    %cc = arith.muli %arg2, %arg2 : i32

    // Combine: (a*b) * (c*c) (depth 2)
    %result = arith.muli %ab, %cc : i32

    secret.yield %result : i32
  } -> !secret.secret<i32>
  return %0 : !secret.secret<i32>
}
```

**Output:**

```mlir
func.func @depth_example(%a: !secret.secret<i32>, %b: !secret.secret<i32>, %c: !secret.secret<i32>) -> !secret.secret<i32> {
  %0 = secret.generic(%a, %b, %c : !secret.secret<i32>, !secret.secret<i32>, !secret.secret<i32>)
      attrs = {arg0 = {mgmt.mgmt = #mgmt.mgmt<level = 2>},
               arg1 = {mgmt.mgmt = #mgmt.mgmt<level = 2>},
               arg2 = {mgmt.mgmt = #mgmt.mgmt<level = 2>}} {
  ^bb0(%arg0: i32, %arg1: i32, %arg2: i32):
    // First multiplication: a * b
    %ab = arith.muli %arg0, %arg1 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : i32
    %1 = mgmt.relinearize %ab {mgmt.mgmt = #mgmt.mgmt<level = 1>} : i32

    // Second multiplication: c * c
    %cc = arith.muli %arg2, %arg2 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : i32
    %2 = mgmt.relinearize %cc {mgmt.mgmt = #mgmt.mgmt<level = 1>} : i32

    // Third multiplication: (a*b) * (c*c) - multiplicative depth 2
    %result = arith.muli %1, %2 {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>} : i32
    %3 = mgmt.relinearize %result {mgmt.mgmt = #mgmt.mgmt<level = 2>} : i32

    secret.yield %3 : i32
  } -> !secret.secret<i32>
  return %0 : !secret.secret<i32>
}
```

## BFV vs BGV Differences

Unlike BGV, BFV typically:

1. **No Modulus Switching**: BFV implementations often don't use modulus
   switching for noise management
1. **Depth Tracking**: The level attribute tracks multiplicative depth rather
   than remaining modulus switches
1. **Relinearization Only**: Only relinearization is inserted, not modulus
   reduction operations
1. **Parameter Selection**: Depth information is used for selecting appropriate
   BFV parameters

## When to Use

The `secret-insert-mgmt-bfv` pass should be used:

1. **Before lowering to BFV backend** to insert necessary relinearization
   operations
1. **For BFV scheme implementations** that require explicit dimension management
1. **To track multiplicative depth** for parameter selection and noise analysis
1. **After secretization and generic distribution** when you have well-formed
   secret operations
1. **In BFV compilation pipelines** as preparation for backend-specific lowering
1. **For noise analysis** where multiplicative depth is a key factor

This pass ensures that BFV ciphertexts remain manageable through proper
relinearization while providing crucial depth information for parameter
selection and noise management.
