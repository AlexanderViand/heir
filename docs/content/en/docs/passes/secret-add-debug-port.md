---
title: Secret Add Debug Port
weight: 17
---

## Overview

The `secret-add-debug-port` pass adds debug instrumentation to secret-arithmetic
operations within `secret.generic` bodies. It inserts calls to debug functions
prefixed with `__heir_debug` after each operation, allowing developers to trace
and inspect intermediate values during encrypted computation.

## Input/Output

- **Input**: IR with `secret.generic` operations containing arithmetic
  operations
- **Output**: IR with debug function calls inserted after each operation, plus
  debug function declarations

## Options

This pass has no configurable options.

## Usage Examples

```bash
heir-opt --secret-add-debug-port input.mlir
```

### Example Input

```mlir
func.func @main(%arg0: !secret.secret<tensor<8xi16>>) -> !secret.secret<tensor<8xi16>> {
  %0 = secret.generic(%arg0 : !secret.secret<tensor<8xi16>>) {
  ^bb0(%arg1: tensor<8xi16>):
    %c100 = arith.constant dense<100> : tensor<8xi16>
    %1 = arith.addi %arg1, %c100 : tensor<8xi16>
    %c2 = arith.constant dense<2> : tensor<8xi16>
    %2 = arith.muli %1, %c2 : tensor<8xi16>
    secret.yield %2 : tensor<8xi16>
  } -> !secret.secret<tensor<8xi16>>
  return %0 : !secret.secret<tensor<8xi16>>
}
```

### Example Output

```mlir
// Debug function declarations added
func.func private @__heir_debug_tensor_8xi16_(tensor<8xi16>)

func.func @main(%arg0: !secret.secret<tensor<8xi16>>) -> !secret.secret<tensor<8xi16>> {
  %0 = secret.generic(%arg0 : !secret.secret<tensor<8xi16>>) {
  ^bb0(%arg1: tensor<8xi16>):
    %c100 = arith.constant dense<100> : tensor<8xi16>
    %1 = arith.addi %arg1, %c100 : tensor<8xi16>
    // Debug call inserted after addition
    func.call @__heir_debug_tensor_8xi16_(%1) : (tensor<8xi16>) -> ()

    %c2 = arith.constant dense<2> : tensor<8xi16>
    %2 = arith.muli %1, %c2 : tensor<8xi16>
    // Debug call inserted after multiplication
    func.call @__heir_debug_tensor_8xi16_(%2) : (tensor<8xi16>) -> ()

    secret.yield %2 : tensor<8xi16>
  } -> !secret.secret<tensor<8xi16>>
  return %0 : !secret.secret<tensor<8xi16>>
}
```

### Complex Example with Mixed Types

**Input:**

```mlir
func.func @mixed_types(%arg0: !secret.secret<i32>, %arg1: !secret.secret<tensor<4xi32>>) -> !secret.secret<tensor<4xi32>> {
  %0 = secret.generic(%arg0, %arg1 : !secret.secret<i32>, !secret.secret<tensor<4xi32>>) {
  ^bb0(%arg2: i32, %arg3: tensor<4xi32>):
    %c5 = arith.constant 5 : i32
    %1 = arith.addi %arg2, %c5 : i32
    %2 = tensor.splat %1 : tensor<4xi32>
    %3 = arith.muli %2, %arg3 : tensor<4xi32>
    secret.yield %3 : tensor<4xi32>
  } -> !secret.secret<tensor<4xi32>>
  return %0 : !secret.secret<tensor<4xi32>>
}
```

**Output:**

```mlir
// Debug function declarations for different types
func.func private @__heir_debug_i32_(i32)
func.func private @__heir_debug_tensor_4xi32_(tensor<4xi32>)

func.func @mixed_types(%arg0: !secret.secret<i32>, %arg1: !secret.secret<tensor<4xi32>>) -> !secret.secret<tensor<4xi32>> {
  %0 = secret.generic(%arg0, %arg1 : !secret.secret<i32>, !secret.secret<tensor<4xi32>>) {
  ^bb0(%arg2: i32, %arg3: tensor<4xi32>):
    %c5 = arith.constant 5 : i32
    %1 = arith.addi %arg2, %c5 : i32
    func.call @__heir_debug_i32_(%1) : (i32) -> ()

    %2 = tensor.splat %1 : tensor<4xi32>
    func.call @__heir_debug_tensor_4xi32_(%2) : (tensor<4xi32>) -> ()

    %3 = arith.muli %2, %arg3 : tensor<4xi32>
    func.call @__heir_debug_tensor_4xi32_(%3) : (tensor<4xi32>) -> ()

    secret.yield %3 : tensor<4xi32>
  } -> !secret.secret<tensor<4xi32>>
  return %0 : !secret.secret<tensor<4xi32>>
}
```

## Debug Function Implementation

Users must provide implementations for the debug functions declared by this
pass. The debug functions follow the naming convention
`__heir_debug_<type_description>_`, where the type description is a mangled
version of the MLIR type.

Example implementations in C++:

```cpp
// For i32 values
extern "C" void __heir_debug_i32_(int32_t value) {
    std::cout << "Debug i32: " << value << std::endl;
}

// For tensor<8xi16> values
extern "C" void __heir_debug_tensor_8xi16_(const std::vector<int16_t>& tensor) {
    std::cout << "Debug tensor<8xi16>: ";
    for (const auto& val : tensor) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}
```

## When to Use

The `secret-add-debug-port` pass should be used:

1. **During development and debugging** to trace intermediate values in
   encrypted computations
1. **For testing and validation** to compare encrypted computation results with
   expected values
1. **For performance analysis** to understand the behavior of secret operations
1. **Before `secret-import-execution-result`** to generate trace data for result
   importation
1. **In conjunction with cleartext backends** to generate reference traces
1. **For educational purposes** to understand the flow of computation in FHE
   programs

This pass is particularly valuable when debugging complex encrypted
computations, as it allows developers to see intermediate results and verify the
correctness of their transformations.
