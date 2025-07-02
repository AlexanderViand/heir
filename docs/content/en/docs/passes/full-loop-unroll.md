---
title: Full Loop Unroll
weight: 10
---

## Overview

The `full-loop-unroll` pass completely unrolls all `affine.for` loops with
constant bounds, expanding each iteration into explicit operations. This
transformation is fundamental for FHE circuit generation and enables aggressive
optimization opportunities in subsequent passes.

## Input/Output

- **Input**: MLIR IR containing `affine.for` loops with statically analyzable
  bounds
- **Output**: MLIR IR with all loops fully unrolled into explicit sequential
  operations

## Options

This pass takes no command-line options.

## Usage Examples

```bash
# Unroll all loops in the IR
heir-opt --full-loop-unroll input.mlir

# Common usage with constant folding
heir-opt --full-loop-unroll --apply-folders input.mlir

# In a complete optimization pipeline
heir-opt --full-loop-unroll --apply-folders --forward-insert-to-extract input.mlir
```

## When to Use

This pass is essential when:

1. **FHE circuit generation**: Before lowering to FHE schemes that require
   explicit, statically-known operations
1. **Enabling constant folding**: To create opportunities for subsequent
   constant folding and simplification passes
1. **Tensor optimization**: Before passes that analyze or optimize tensor access
   patterns
1. **Static analysis**: When downstream passes require explicit operation
   sequences rather than loops
1. **Memory optimization**: Before passes that optimize memory access patterns
   revealed by unrolling

The pass is commonly used early in optimization pipelines, especially before
`apply-folders` and memory optimization passes.

## Implementation Details

The transformation process:

1. **Loop Detection**: Identifies all `affine.for` loops in the IR
1. **Bound Analysis**: Verifies that loop bounds are statically analyzable
   constants
1. **Iteration Expansion**: Expands each loop iteration into explicit operations
1. **SSA Value Mapping**: Creates unique SSA values for each unrolled iteration
1. **Control Flow Elimination**: Removes loop control flow constructs

**Requirements and Limitations:**

- Only operates on `affine.for` loops with constant bounds
- Loop bounds must be statically analyzable at compile time
- Very large loops may cause excessive code expansion
- Does not handle loops with dynamic or symbolic bounds

**Benefits:**

- Eliminates loop overhead completely
- Enables aggressive constant propagation and folding
- Reveals optimization opportunities in memory access patterns
- Simplifies control flow for downstream analysis passes
- Essential for generating efficient FHE circuits

**Performance Considerations:**

- Can significantly increase IR size for large loops
- May increase compilation time due to larger IR
- Generally improves runtime performance by eliminating loop overhead
- Enables better optimization in subsequent passes
