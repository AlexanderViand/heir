---
title: Convert If to Select
weight: 8
---

## Overview

The `convert-if-to-select` pass converts `scf.if` operations with
secret-dependent conditions into `arith.select` operations. This transformation
eliminates branching based on secret values, ensuring constant-time execution
and completing the data-oblivious transformation pipeline.

## Input/Output

- **Input**: MLIR IR containing `scf.if` operations with secret-dependent
  conditions
- **Output**: MLIR IR with `arith.select` operations that evaluate both branches
  and select results without branching

## Options

This pass takes no command-line options.

## Usage Examples

```bash
# Convert secret-dependent conditionals to select operations
heir-opt --convert-if-to-select input.mlir

# Common usage as final step in data-oblivious pipeline
heir-opt --convert-secret-while-to-static-for --convert-secret-for-to-static-for --convert-secret-extract-to-static-extract --convert-if-to-select input.mlir
```

## When to Use

This pass is the final step when:

1. **Completing data-oblivious transformation**: After other passes have
   converted loops and memory access patterns
1. **Eliminating secret-dependent branching**: When all conditional operations
   based on secret values must be removed
1. **FHE circuit generation**: Before lowering to FHE schemes that cannot handle
   data-dependent control flow
1. **Side-channel resistance**: When execution flow must not leak information
   about secret inputs
1. **Constant-time guarantees**: When programs must execute in the same time
   regardless of input values

This pass is typically the last transformation in a data-oblivious compilation
pipeline.

## Implementation Details

The transformation process:

1. **Condition Analysis**: Identifies `scf.if` operations with secret-dependent
   conditions
1. **Branch Evaluation**: Ensures both `then` and `else` branches are evaluated
1. **Result Selection**: Uses `arith.select` to choose between branch results
   based on the condition
1. **Multi-value Handling**: Creates multiple select operations for `scf.if`
   operations that yield multiple values
1. **SSA Preservation**: Maintains proper SSA form throughout the transformation

**Security Properties:**

- Eliminates all data-dependent branching
- Ensures both code paths execute regardless of condition
- Provides constant-time execution guarantees
- Prevents information leakage through execution flow

**Performance Implications:**

- Both branches of every conditional are executed
- Computational cost increases but execution time becomes predictable
- Memory access patterns become uniform
- Enables aggressive compiler optimizations on the resulting linear code
