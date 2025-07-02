---
title: Convert Secret Insert to Static Insert
weight: 9
---

## Overview

The `convert-secret-insert-to-static-insert` pass converts `tensor.insert`
operations that use secret indices into data-oblivious operations that perform
insertions at all possible indices and conditionally select the correct result
tensor. This transformation prevents information leakage through memory write
patterns.

## Input/Output

- **Input**: MLIR IR containing `tensor.insert` operations with secret-dependent
  indices
- **Output**: MLIR IR with static insertion loops that attempt writes at all
  indices and conditionally select tensors using `scf.if` operations

## Options

This pass takes no command-line options.

## Usage Examples

```bash
# Convert secret-indexed insertions to static insertions
heir-opt --convert-secret-insert-to-static-insert input.mlir

# Common usage in data-oblivious pipeline with extract operations
heir-opt --convert-secret-extract-to-static-extract --convert-secret-insert-to-static-insert --convert-if-to-select input.mlir
```

## When to Use

This pass is essential when:

1. **Preventing write-pattern attacks**: When memory write patterns must not
   reveal information about secret indices
1. **Complementing extract transformations**: Alongside
   `convert-secret-extract-to-static-extract` for complete memory operation
   safety
1. **FHE circuit generation**: Before lowering to FHE schemes where all memory
   operations must be statically determinable
1. **Data-oblivious algorithms**: When implementing algorithms that must
   maintain uniform memory access patterns
1. **Security-critical applications**: When side-channel resistance through
   uniform write patterns is required

The pass must be used in conjunction with `convert-if-to-select` to achieve
complete data-oblivious execution.

## Implementation Details

The transformation process:

1. **Secret Index Detection**: Identifies `tensor.insert` operations using
   secret indices
1. **Loop Generation**: Creates `affine.for` loops that iterate through all
   possible indices
1. **Universal Insertion**: Performs insertions at each possible index position
1. **Conditional Selection**: Uses `scf.if` operations to select the tensor with
   the insert at the secret index
1. **Tensor Propagation**: Maintains proper tensor flow through `iter_args`
   while preserving immutability

**Performance Considerations:**

- Time complexity changes from O(1) to O(n) where n is the tensor dimension
- Creates intermediate tensors for each potential insertion point
- All tensor positions are potentially modified regardless of the secret index
- Memory usage increases due to intermediate tensor creation

**Security Benefits:**

- Eliminates data-dependent memory write patterns
- Prevents timing-based side-channel attacks through uniform access
- Ensures constant execution time regardless of secret values
- Complements extract operations to provide complete memory operation security

**Memory Model:**

- Preserves tensor immutability semantics
- Each iteration creates a new tensor through conditional selection
- Final result is the tensor with the value inserted at the secret index
