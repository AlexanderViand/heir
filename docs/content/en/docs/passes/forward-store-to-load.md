---
title: Forward Store to Load
weight: 11
---

## Overview

The `forward-store-to-load` pass optimizes memory operations by forwarding
stored values directly to subsequent loads, eliminating redundant memory
traffic. This pass acts as a simplified mem2reg transformation focusing on
single-block scenarios where store-load pairs can be directly eliminated.

## Input/Output

- **Input**: MLIR IR containing `affine.store`/`affine.load` or
  `memref.store`/`memref.load` operations within single basic blocks
- **Output**: MLIR IR with eliminated store-load pairs where loads are replaced
  with direct value usage

## Options

This pass takes no command-line options.

## Usage Examples

```bash
# Forward stores to loads within basic blocks
heir-opt --forward-store-to-load input.mlir

# Common usage after loop unrolling
heir-opt --full-loop-unroll --forward-store-to-load input.mlir

# In optimization pipeline with constant folding
heir-opt --full-loop-unroll --forward-store-to-load --apply-folders input.mlir
```

## When to Use

This pass is beneficial when:

1. **After loop unrolling**: When unrolled loops create clear store-load
   patterns that can be optimized
1. **Memory optimization**: To reduce memory traffic by eliminating unnecessary
   round-trips to memory
1. **Scalar replacement**: As part of converting memory operations to direct
   value operations
1. **Preprocessing for further optimization**: Before passes that benefit from
   reduced memory operations
1. **FHE circuit optimization**: To simplify memory operations before lowering
   to FHE schemes

The pass is commonly used after `full-loop-unroll` when memory access patterns
become statically analyzable.

## Implementation Details

The transformation process:

1. **Block Analysis**: Analyzes each basic block independently for store-load
   pairs
1. **Memory Access Analysis**: Identifies affine and memref operations that
   access the same memory locations
1. **Forwarding Opportunities**: Detects stores followed by loads of the same
   memory location
1. **Value Replacement**: Replaces load operations with direct use of stored
   values
1. **Dead Store Elimination**: Removes stores that are immediately forwarded and
   have no further uses

**Scope and Limitations:**

- **Single Block Only**: Only analyzes within individual basic blocks
- **No Control Flow**: Does not handle complex control flow or inter-block
  analysis
- **Static Analysis**: Requires statically analyzable memory access patterns
- **Affine Preferred**: Works best with affine operations that have clear access
  patterns

**Benefits:**

- Reduces memory traffic and improves performance
- Simplifies IR for subsequent optimization passes
- Enables scalar replacement of memory operations
- Creates opportunities for constant propagation and folding

**Pipeline Integration:**

- Typically used after loop unrolling reveals access patterns
- Often followed by constant folding and dead code elimination
- Complements other memory optimization passes
