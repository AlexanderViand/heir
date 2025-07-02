---
title: Annotate Secretness
weight: 2
---

## Overview

The `annotate-secretness` pass is a debugging helper that runs secretness
analysis and annotates the IR with the results. It extends the `{secret.secret}`
annotation to all operation results, function arguments, return types, and
terminators that are determined to be secret.

## Input/Output

- **Input**: MLIR IR with secret dialect operations and potential secret values
- **Output**: Same IR with comprehensive secretness annotations on all relevant
  values

## Options

| Option    | Type | Default | Description                                                                                                                                                            |
| --------- | ---- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `verbose` | bool | `false` | If true, annotate all values including public ones with `{secret.public}`, missing analysis with `{secret.missing}`, and inconclusive analysis with `{secret.unknown}` |

## Usage Examples

```bash
# Basic secretness annotation
heir-opt --annotate-secretness input.mlir

# Verbose mode showing all secretness states
heir-opt --annotate-secretness='verbose=true' input.mlir
```

### Example Input/Output

**Input:**

```mlir
func.func @example(%s: i32 {secret.secret}, %p: i32) -> i32 {
  %0 = arith.addi %p, %p : i32
  %1 = arith.addi %s, %p : i32
  return %1 : i32
}
```

**Output (verbose mode):**

```mlir
func.func @example(%s: i32 {secret.secret}, %p: i32 {secret.public}) -> (i32 {secret.secret}) {
  %0 = arith.addi %p, %p {secret.public} : i32
  %1 = arith.addi %s, %p {secret.secret} : i32
  return {secret.secret} %1 : i32
}
```

## When to Use

This pass is primarily used for:

1. **Debugging secretness analysis**: Understanding how secret values propagate
   through the IR
1. **Verification**: Ensuring that secretness analysis produces expected results
1. **Development**: When implementing new transformations that depend on
   secretness
1. **Pipeline validation**: Confirming that secret handling is correct before
   lowering

The pass should be used during development and debugging phases, not in
production compilation pipelines, as the annotations are primarily for human
inspection.

## Implementation Details

The pass:

- Runs the full secretness analysis using MLIR's DataFlow framework
- Annotates operation results, function arguments, return types, and terminators
- Supports multi-result operations with per-result secretness annotations
- Handles secret dialect operations and generic secret computations
