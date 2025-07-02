---
title: Secret Import Execution Result
weight: 18
---

## Overview

The `secret-import-execution-result` pass imports execution results from a trace
file and adds them as `secret.execution_result` attributes to secret-arithmetic
operations. This is useful for comparing precision between plaintext and
ciphertext computations, particularly for schemes like CKKS that involve
approximate arithmetic.

## Input/Output

- **Input**: IR with secret-arithmetic operations and a trace file containing
  execution results
- **Output**: IR with `secret.execution_result` attributes added to operations,
  containing the imported trace data

## Options

- `--file-name=<path>`: Path to the trace file containing execution results.
  Each line in the file corresponds to one SSA value in the IR.

## Usage Examples

```bash
heir-opt --secret-import-execution-result="file-name=trace.log" input.mlir
```

### Trace File Format

The trace file should contain one line per SSA value, with space-separated
numerical values:

**trace.log:**

```
1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0
2.0 4.0 6.0 8.0 10.0 12.0 14.0 16.0
4.0 8.0 12.0 16.0 20.0 24.0 28.0 32.0
```

### Example Input

```mlir
func.func @main(%arg0: !secret.secret<tensor<8xf64>>) -> !secret.secret<tensor<8xf64>> {
  %0 = secret.generic(%arg0 : !secret.secret<tensor<8xf64>>) {
  ^bb0(%arg1: tensor<8xf64>):
    %c2 = arith.constant dense<2.0> : tensor<8xf64>
    %1 = arith.mulf %arg1, %c2 : tensor<8xf64>
    %2 = arith.mulf %1, %c2 : tensor<8xf64>
    secret.yield %2 : tensor<8xf64>
  } -> !secret.secret<tensor<8xf64>>
  return %0 : !secret.secret<tensor<8xf64>>
}
```

### Example Output

```mlir
func.func @main(%arg0: !secret.secret<tensor<8xf64>>) -> !secret.secret<tensor<8xf64>> {
  %0 = secret.generic(%arg0 : !secret.secret<tensor<8xf64>>) {
  ^bb0(%arg1: tensor<8xf64>):
    %c2 = arith.constant dense<2.0> : tensor<8xf64>
    %1 = arith.mulf %arg1, %c2 {secret.execution_result = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]} : tensor<8xf64>
    %2 = arith.mulf %1, %c2 {secret.execution_result = [4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0]} : tensor<8xf64>
    secret.yield %2 : tensor<8xf64>
  } -> !secret.secret<tensor<8xf64>>
  return %0 : !secret.secret<tensor<8xf64>>
}
```

### CKKS Precision Analysis Example

This pass is particularly useful for CKKS precision analysis:

**Input with CKKS operations:**

```mlir
func.func @ckks_precision(%arg0: !secret.secret<tensor<4xf64>>) -> !secret.secret<tensor<4xf64>> {
  %0 = secret.generic(%arg0 : !secret.secret<tensor<4xf64>>) {
  ^bb0(%arg1: tensor<4xf64>):
    // Polynomial evaluation: x^2 + 2x + 1
    %c1 = arith.constant dense<1.0> : tensor<4xf64>
    %c2 = arith.constant dense<2.0> : tensor<4xf64>

    %x_squared = arith.mulf %arg1, %arg1 : tensor<4xf64>
    %two_x = arith.mulf %arg1, %c2 : tensor<4xf64>
    %result1 = arith.addf %x_squared, %two_x : tensor<4xf64>
    %result2 = arith.addf %result1, %c1 : tensor<4xf64>

    secret.yield %result2 : tensor<4xf64>
  } -> !secret.secret<tensor<4xf64>>
  return %0 : !secret.secret<tensor<4xf64>>
}
```

**Trace file (plaintext_trace.log):**

```
1.0 4.0 9.0 16.0
2.0 4.0 6.0 8.0
3.0 8.0 15.0 24.0
4.0 9.0 16.0 25.0
```

**Output with imported results:**

```mlir
func.func @ckks_precision(%arg0: !secret.secret<tensor<4xf64>>) -> !secret.secret<tensor<4xf64>> {
  %0 = secret.generic(%arg0 : !secret.secret<tensor<4xf64>>) {
  ^bb0(%arg1: tensor<4xf64>):
    %c1 = arith.constant dense<1.0> : tensor<4xf64>
    %c2 = arith.constant dense<2.0> : tensor<4xf64>

    %x_squared = arith.mulf %arg1, %arg1 {secret.execution_result = [1.0, 4.0, 9.0, 16.0]} : tensor<4xf64>
    %two_x = arith.mulf %arg1, %c2 {secret.execution_result = [2.0, 4.0, 6.0, 8.0]} : tensor<4xf64>
    %result1 = arith.addf %x_squared, %two_x {secret.execution_result = [3.0, 8.0, 15.0, 24.0]} : tensor<4xf64>
    %result2 = arith.addf %result1, %c1 {secret.execution_result = [4.0, 9.0, 16.0, 25.0]} : tensor<4xf64>

    secret.yield %result2 : tensor<4xf64>
  } -> !secret.secret<tensor<4xf64>>
  return %0 : !secret.secret<tensor<4xf64>>
}
```

## Workflow Integration

This pass is typically used in a workflow like:

1. Run the program with `--secret-add-debug-port` using a plaintext backend
1. Capture the debug output to a trace file
1. Run `--secret-import-execution-result` to import the trace data
1. Run the same program with an FHE backend and compare results

## When to Use

The `secret-import-execution-result` pass should be used:

1. **For precision analysis** comparing plaintext and ciphertext computation
   results
1. **With CKKS schemes** where approximate arithmetic may introduce precision
   loss
1. **After running with `secret-add-debug-port`** to import the generated trace
   data
1. **For debugging** to understand expected vs. actual computation results
1. **In testing workflows** to verify correctness of FHE implementations
1. **For performance analysis** to understand numerical behavior of encrypted
   operations

This pass is essential for validating the numerical correctness of FHE
computations, especially when working with approximate schemes where precision
loss can occur.
