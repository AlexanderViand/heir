---
title: Forward Insert to Extract
weight: 12
---

## Overview

The `forward-insert-to-extract` pass optimizes tensor operations by forwarding
inserted values directly to subsequent extractions, eliminating redundant tensor
operations. This is the tensor equivalent of `forward-store-to-load`, designed
for MLIR's immutable tensor semantics.

## Input/Output

- **Input**: MLIR IR containing `tensor.insert` and `tensor.extract` operations
  within single basic blocks
- **Output**: MLIR IR with eliminated insert-extract pairs where extracts are
  replaced with direct value usage

## Options

This pass takes no command-line options.

## Usage Examples

```bash
# Forward tensor inserts to extracts
heir-opt --forward-insert-to-extract input.mlir

# Common usage after loop unrolling
heir-opt --full-loop-unroll --forward-insert-to-extract input.mlir

# In comprehensive tensor optimization pipeline
heir-opt --full-loop-unroll --forward-insert-to-extract --apply-folders input.mlir
```

## When to Use

This pass is particularly effective when:

1. **After loop unrolling**: When unrolled loops create predictable tensor
   insert-extract patterns
1. **Tensor optimization**: To reduce tensor operation overhead by eliminating
   redundant operations
1. **FHE tensor processing**: Before lowering tensor operations to FHE-specific
   representations
1. **Preprocessing for constant folding**: Creating opportunities for subsequent
   optimization passes
1. **Immutable data structure optimization**: When working with functional-style
   tensor transformations

The pass is commonly used after `full-loop-unroll` and often paired with
`apply-folders` for comprehensive tensor optimization.

## Implementation Details

The transformation process:

1. **Single-Block Analysis**: Analyzes tensor operations within individual basic
   blocks
1. **Insert-Extract Matching**: Identifies `tensor.insert` operations followed
   by `tensor.extract` operations at the same indices
1. **Value Forwarding**: Replaces extract operations with direct use of the
   inserted values
1. **Tensor Chain Optimization**: Maintains tensor immutability while
   eliminating unnecessary intermediate operations

**Scope and Limitations:**

- **Single Block Only**: Analysis is limited to individual basic blocks
- **No Control Flow**: Does not handle complex control flow or inter-block
  dependencies
- **Static Index Matching**: Requires statically analyzable index expressions
  for forwarding
- **Immutable Semantics**: Preserves tensor immutability guarantees

**Benefits:**

- Reduces tensor creation and access overhead
- Eliminates redundant tensor operations
- Simplifies tensor computation chains
- Creates opportunities for constant propagation in tensor contexts
- Maintains functional programming semantics while improving performance

**Integration Patterns:**

- Typically follows loop unrolling that exposes tensor access patterns
- Often paired with constant folding passes for maximum benefit
- Complements other tensor optimization transformations
- Useful in FHE pipelines where tensor operations are performance-critical
