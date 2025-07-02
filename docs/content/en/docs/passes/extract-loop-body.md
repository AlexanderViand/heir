---
title: Extract Loop Body
weight: 15
---

## Overview

The `extract-loop-body` pass extracts the computational logic inside loop bodies
into separate functions. This transformation is useful for creating modular,
reusable functions from loop iterations and can simplify subsequent analysis and
optimization phases.

## Input/Output

- **Input**: MLIR module with nested affine loops containing computational logic
- **Output**: Same module with loop bodies extracted into separate functions and
  loops calling these extracted functions

## Options

- `--min-loop-size` (unsigned, default: 4): Minimum iteration count for a loop
  to be considered for extraction
- `--min-body-size` (unsigned, default: 4): Minimum number of operations in a
  loop body to justify extraction

## Usage Examples

```bash
# Extract loop bodies with default thresholds
heir-opt --extract-loop-body input.mlir

# Use custom thresholds for extraction
heir-opt --extract-loop-body --min-loop-size=8 --min-body-size=6 input.mlir

# Extract only larger loop bodies
heir-opt --extract-loop-body --min-loop-size=16 --min-body-size=10 input.mlir
```

## When to Use

This pass should be used when:

1. **Creating Reusable Functions**: Extract repeated computational patterns from
   loops
1. **Simplifying Analysis**: Break complex nested loops into simpler function
   calls
1. **Preparing for Optimization**: Create functions that can be optimized
   independently
1. **Code Organization**: Improve code structure by separating iteration logic
   from computation

Typical use cases:

- Neural network layer implementations with repeated operations
- Image processing kernels with per-pixel computations
- Mathematical computations with regular iteration patterns

## Extraction Criteria

The pass extracts loop bodies when:

- Loop has at least `min-loop-size` iterations
- Loop body contains at least `min-body-size` operations
- Loop body has a clear input/output pattern (loads as inputs, single store as
  output)
- Loop body doesn't contain complex control flow

## Function Generation

The extracted function:

- Takes loaded values as parameters
- Returns the value that would be stored
- Preserves the computational logic exactly
- Uses a generated name (e.g., `__for_loop`, `__for_loop_0`, etc.)

## Example Transformation

**Input:**

```mlir
module {
  func.func @image_process(%input: memref<25x20x8xi8>) -> memref<25x20x8xi8> {
    %c-128_i8 = arith.constant -128 : i8
    %c127_i8 = arith.constant 127 : i8
    %output = memref.alloc() : memref<25x20x8xi8>

    affine.for %i = 0 to 25 {
      affine.for %j = 0 to 20 {
        affine.for %k = 0 to 8 {
          %pixel = affine.load %input[%i, %j, %k] : memref<25x20x8xi8>
          // Complex pixel processing
          %processed = arith.muli %pixel, %pixel : i8
          %shifted = arith.shli %processed, %c2_i8 : i8
          %clamped_low = arith.cmpi slt, %shifted, %c-128_i8 : i8
          %temp1 = arith.select %clamped_low, %c-128_i8, %shifted : i8
          %clamped_high = arith.cmpi sgt, %temp1, %c127_i8 : i8
          %final = arith.select %clamped_high, %c127_i8, %temp1 : i8
          affine.store %final, %output[%i, %j, %k] : memref<25x20x8xi8>
        }
      }
    }
    return %output : memref<25x20x8xi8>
  }
}
```

**Output:**

```mlir
module {
  func.func @image_process(%input: memref<25x20x8xi8>) -> memref<25x20x8xi8> {
    %output = memref.alloc() : memref<25x20x8xi8>

    affine.for %i = 0 to 25 {
      affine.for %j = 0 to 20 {
        affine.for %k = 0 to 8 {
          %pixel = affine.load %input[%i, %j, %k] : memref<25x20x8xi8>
          %result = func.call @__for_loop(%pixel) : (i8) -> i8
          affine.store %result, %output[%i, %j, %k] : memref<25x20x8xi8>
        }
      }
    }
    return %output : memref<25x20x8xi8>
  }

  // Extracted function containing the computation logic
  func.func private @__for_loop(%arg0: i8) -> i8 {
    %c-128_i8 = arith.constant -128 : i8
    %c127_i8 = arith.constant 127 : i8
    %c2_i8 = arith.constant 2 : i8

    %processed = arith.muli %arg0, %arg0 : i8
    %shifted = arith.shli %processed, %c2_i8 : i8
    %clamped_low = arith.cmpi slt, %shifted, %c-128_i8 : i8
    %temp1 = arith.select %clamped_low, %c-128_i8, %shifted : i8
    %clamped_high = arith.cmpi sgt, %temp1, %c127_i8 : i8
    %final = arith.select %clamped_high, %c127_i8, %temp1 : i8

    return %final : i8
  }
}
```

## Benefits

### Code Organization

- **Separation of Concerns**: Iteration logic separated from computation logic
- **Reusability**: Extracted functions can potentially be reused
- **Testability**: Individual computational kernels can be tested independently

### Analysis and Optimization

- **Function-Level Optimization**: Each extracted function can be optimized
  independently
- **Simplified Loop Analysis**: Loops become simpler with single function calls
- **Better Inlining Decisions**: Compiler can make informed decisions about
  inlining

### Debugging and Profiling

- **Granular Profiling**: Profile individual computational kernels
- **Easier Debugging**: Set breakpoints on specific computational logic
- **Code Coverage**: Better understanding of which computations are exercised

## Prerequisites

- Input must use affine dialect for loop operations
- Loop bodies should follow load-compute-store pattern
- Memref operations must use static indexing
- Complex control flow within loop bodies may prevent extraction

## Integration Notes

This pass works well with:

- **Function Optimization**: Apply function-level passes to extracted functions
- **Inlining**: Later inlining passes can reintegrate functions if beneficial
- **Vectorization**: Extracted functions may be good candidates for
  vectorization
