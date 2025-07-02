---
title: Apply Folders
weight: 1
---

## Overview

The `apply-folders` pass applies all registered folding patterns greedily to the
input IR. This pass focuses specifically on folding operations without running
the full canonicalization pipeline, making it more efficient when only constant
folding and similar simplifications are needed.

## Input/Output

- **Input**: MLIR IR with operations that can benefit from folding patterns
- **Output**: MLIR IR with folded constants and simplified operations,
  particularly tensor extract slice operations

## Options

This pass takes no command-line options.

## Usage Examples

```bash
# Apply folding patterns to simplify the IR
heir-opt --apply-folders input.mlir

# Common usage in pipeline after loop unrolling
heir-opt --full-loop-unroll --apply-folders input.mlir
```

## When to Use

This pass is particularly useful:

1. **After loop unrolling**: When `full-loop-unroll` creates many constant
   operations that can be folded
1. **Before insert-rotate optimization**: As a preprocessing step to simplify IR
   before rotation analysis
1. **When canonicalize is too slow**: As a lighter-weight alternative when only
   folding is needed
1. **Pipeline optimization**: To prepare IR for subsequent passes that benefit
   from simplified constants

The pass is commonly used in tensor-heavy computations where constant folding of
tensor extract slice operations provides significant simplification without the
overhead of full canonicalization.

## Implementation Details

The pass specifically focuses on:

- Constant folding of tensor extract slice operations
- Greedy application of folding patterns
- Lightweight simplification compared to full canonicalization
