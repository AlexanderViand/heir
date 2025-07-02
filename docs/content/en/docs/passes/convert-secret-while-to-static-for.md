---
title: Convert Secret While to Static For
weight: 5
---

## Overview

The `convert-secret-while-to-static-for` pass converts `scf.while` loops with
secret-dependent conditions to `affine.for` loops with constant bounds. This
transformation is essential for creating data-oblivious programs where loop
iteration counts cannot depend on secret values.

## Input/Output

- **Input**: MLIR IR with `scf.while` loops that have secret-dependent
  conditions
- **Output**: MLIR IR with `affine.for` loops with constant bounds and
  conditional execution inside the loop body

## Requirements

- The `scf.while` operation must have a `max_iter` attribute specifying the
  maximum number of iterations
- The loop condition must be secret-dependent (involving secret values)

## Usage Examples

```bash
# Convert secret while loops to static for loops
heir-opt --convert-secret-while-to-static-for input.mlir

# Often used with convert-if-to-select for full data-oblivious transformation
heir-opt --convert-secret-while-to-static-for --convert-if-to-select input.mlir
```

### Example Transformation

**Input:**

```mlir
func.func @main(%secretInput: !secret.secret<i16>) -> !secret.secret<i16> {
  %c100 = arith.constant 100 : i16
  %0 = secret.generic(%secretInput : !secret.secret<i16>) {
  ^bb0(%input: i16):
    %1 = scf.while (%arg1 = %input) : (i16) -> i16 {
      %2 = arith.cmpi sgt, %arg1, %c100 : i16
      scf.condition(%2) %arg1 : i16
    } do {
    ^bb0(%arg1: i16):
      %3 = arith.muli %arg1, %arg1 : i16
      scf.yield %3 : i16
    } attributes {max_iter = 16 : i64}
    secret.yield %1 : i16
  } -> !secret.secret<i16>
  return %0 : !secret.secret<i16>
}
```

**Output:**

```mlir
func.func @main(%secretInput: !secret.secret<i16>) -> !secret.secret<i16> {
  %c100 = arith.constant 100 : i16
  %0 = secret.generic(%secretInput : !secret.secret<i16>) {
  ^bb0(%input: i16):
    %1 = affine.for %i = 0 to 16 iter_args(%arg1 = %input) -> (i16) {
      %2 = arith.cmpi sgt, %arg1, %c100 : i16
      %3 = scf.if %2 -> i16 {
        %4 = arith.muli %arg1, %arg1 : i16
        scf.yield %4 : i16
      } else {
        scf.yield %arg1 : i16
      }
      affine.yield %3 : i16
    }
    secret.yield %1 : i16
  } -> !secret.secret<i16>
  return %0 : !secret.secret<i16>
}
```

## When to Use

This pass is essential for:

1. **Data-oblivious compilation**: Converting secret-dependent control flow to
   fixed control flow
1. **FHE preparation**: Ensuring loop bounds are known at compile time for FHE
   circuits
1. **Security analysis**: Making timing and access patterns independent of
   secret values
1. **Circuit compilation**: Preparing for compilation to fixed-size boolean or
   arithmetic circuits

## Important Notes

### Security Considerations

- This pass alone does **not** create a fully data-oblivious program
- Must be followed by `--convert-if-to-select` to handle the conditional
  execution
- The maximum iteration count must be chosen carefully to ensure correctness

### Performance Impact

- May increase computation time if `max_iter` is much larger than typical
  iteration counts
- Trade-off between security (data-obliviousness) and performance
- Critical for FHE where data-dependent control flow is not feasible

## Implementation Details

The transformation:

1. Replaces `scf.while` with `affine.for` using the `max_iter` bound
1. Moves the loop condition inside the loop body as an `scf.if`
1. Uses conditional execution to maintain correct semantics
1. Preserves loop-carried values through `iter_args`

## Related Passes

- `convert-if-to-select`: Completes the data-oblivious transformation
- `convert-secret-for-to-static-for`: Similar transformation for `scf.for` loops
- `full-loop-unroll`: Alternative for small, known iteration counts
