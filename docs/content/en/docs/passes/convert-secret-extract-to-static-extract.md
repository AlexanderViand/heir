---
title: Convert Secret Extract to Static Extract
weight: 7
---

## Overview

The `convert-secret-extract-to-static-extract` pass converts `tensor.extract`
operations that use secret indices into data-oblivious operations that extract
all possible values and conditionally select the correct one. This
transformation is essential for preventing information leakage through memory
access patterns.

## Input/Output

- **Input**: MLIR IR containing `tensor.extract` operations with
  secret-dependent indices
- **Output**: MLIR IR with static extraction loops that access all indices and
  conditionally select values using `scf.if` operations

## Options

This pass takes no command-line options.

## Usage Examples

```bash
# Convert secret-indexed extractions to static extractions
heir-opt --convert-secret-extract-to-static-extract input.mlir

# Common usage in data-oblivious pipeline
heir-opt --convert-secret-extract-to-static-extract --convert-if-to-select input.mlir
```

## When to Use

This pass is essential when:

1. **Preventing timing attacks**: When memory access patterns must not reveal
   information about secret indices
1. **FHE circuit generation**: Before lowering to FHE schemes where all memory
   accesses must be known at compile time
1. **Data-oblivious algorithms**: When implementing algorithms that must
   maintain constant-time execution
1. **Security-critical applications**: When side-channel resistance through
   uniform memory access is required

The pass must be used in conjunction with `convert-if-to-select` to achieve
complete data-oblivious execution.

## Implementation Details

The transformation works by:

1. **Secret Index Detection**: Identifies `tensor.extract` operations using
   secret indices
1. **Loop Generation**: Creates `affine.for` loops that iterate through all
   possible indices
1. **Universal Extraction**: Extracts values at each possible index position
1. **Conditional Selection**: Uses `scf.if` operations to select the value from
   the secret index
1. **Value Propagation**: Maintains proper data flow through `iter_args`

**Performance Considerations:**

- Time complexity changes from O(1) to O(n) where n is the tensor dimension
- All tensor elements are accessed regardless of the secret index
- Memory access patterns become uniform and predictable

**Security Benefits:**

- Eliminates data-dependent memory access patterns
- Prevents timing-based side-channel attacks
- Ensures constant execution time regardless of secret values
