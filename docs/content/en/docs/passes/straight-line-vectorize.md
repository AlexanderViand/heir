---
title: Straight Line Vectorizer
weight: 3
---

## Overview

The straight line vectorizer is a simple vectorization pass that operates on
straight-line programs (code without control flow) within a given region. This
pass focuses on vectorizing sequences of operations that can be executed in
parallel.

## Input/Output

- **Input**: IR containing straight-line programs with operations that can be
  vectorized
- **Output**: Vectorized IR where suitable operations have been converted to
  work on vector/tensor types

## Options

- `--dialect=<string>`: Restrict vectorization to operations from the specified
  dialect (default: empty, vectorizes all suitable operations)

## Usage Examples

```bash
heir-opt --straight-line-vectorize input.mlir
```

Restrict to a specific dialect:

```bash
heir-opt --straight-line-vectorize=dialect=arith input.mlir
```

## When to Use

Use this pass when you have:

1. Straight-line code (no branches, loops, or complex control flow)
1. Operations that can benefit from vectorization
1. IR where you want to exploit SIMD parallelism
1. Code that will be lowered to vector-capable backends

Typical placement in compilation pipelines:

1. After initial IR generation
1. Before more complex vectorization passes
1. Early in the pipeline, before dialect-specific lowering

## How It Works

The pass operates by:

1. Analyzing regions of straight-line code
1. Identifying operations that can be safely vectorized
1. Converting scalar operations to vector/tensor operations where beneficial
1. Respecting dialect restrictions when specified
1. Ignoring control flow constructs (branches, loops)

## Example

**Before vectorization:**

```mlir
func.func @straight_line(%a: f32, %b: f32, %c: f32, %d: f32) -> (f32, f32) {
  %0 = arith.addf %a, %b : f32
  %1 = arith.mulf %c, %d : f32
  %2 = arith.addf %0, %1 : f32
  %3 = arith.mulf %0, %1 : f32
  return %2, %3 : f32, f32
}
```

**After vectorization (conceptual example):**

```mlir
func.func @straight_line(%a: f32, %b: f32, %c: f32, %d: f32) -> (f32, f32) {
  %vec_input = tensor.from_elements %a, %c : tensor<2xf32>
  %vec_input2 = tensor.from_elements %b, %d : tensor<2xf32>
  %vec_result = arith.addf %vec_input, %vec_input2 : tensor<2xf32>
  %0 = tensor.extract %vec_result[0] : tensor<2xf32>
  %1 = tensor.extract %vec_result[1] : tensor<2xf32>
  %2 = arith.addf %0, %1 : f32
  %3 = arith.mulf %0, %1 : f32
  return %2, %3 : f32, f32
}
```

## Limitations

- Only operates on straight-line programs
- Does not handle control flow (branches, loops)
- Effectiveness depends on the availability of vectorizable operations
- May not always be beneficial for small-scale parallelism

## Related Passes

- Use before `insert-rotate` for FHE-specific vectorization
- Can be combined with `--canonicalize` to clean up the result
- Works well with layout propagation passes in FHE pipelines
