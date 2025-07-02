---
title: Collapse Insertion Chains
weight: 5
---

## Overview

The collapse insertion chains pass is a cleanup pass that identifies and
optimizes chains of extract/insert operations by replacing them with single
rotation operations when possible. This pass is specifically designed to clean
up patterns left behind by the `insert-rotate` pass.

## Input/Output

- **Input**: IR containing chains of `tensor.extract` and `tensor.insert`
  operations with constant indices
- **Output**: Optimized IR where eligible insertion chains are replaced with
  `tensor_ext.rotate` operations

## Options

This pass has no command-line options.

## Usage Examples

```bash
heir-opt --collapse-insertion-chains input.mlir
```

Typically used as part of a cleanup pipeline:

```bash
heir-opt --insert-rotate --cse --canonicalize --collapse-insertion-chains input.mlir
```

## When to Use

Use this pass as a cleanup step:

1. **After `insert-rotate`** - Primary use case to clean up insertion patterns
1. **After canonicalization passes** - Ensures constant indices are folded
1. **Before further optimizations** - Simplifies IR for downstream passes

Requirements for effective operation:

- Constant indices (may require `--canonicalize` or `--sccp` first)
- Complete insertion chains (all tensor indices must be written)
- Consistent extraction patterns

## How It Works

The pass operates by:

1. **Pattern Detection**: Identifies chains of `tensor.insert` operations
1. **Shift Analysis**: Calculates consistent offset between extraction and
   insertion indices
1. **Chain Validation**: Ensures all tensor indices are covered and come from
   the same source
1. **Replacement**: Converts complete chains to single `tensor_ext.rotate`
   operations

### Requirements for Collapsing

A chain can be collapsed if:

- All insertions use constant indices
- All extractions come from the same source tensor
- The shift pattern is consistent across all operations
- Every index of the destination tensor is written
- Source and destination tensors have the same shape

## Example

**Before collapse-insertion-chains:**

```mlir
%extracted = tensor.extract %source[%c5] : tensor<16xi16>
%inserted = tensor.insert %extracted into %dest[%c0] : tensor<16xi16>
%extracted_0 = tensor.extract %source[%c6] : tensor<16xi16>
%inserted_1 = tensor.insert %extracted_0 into %inserted[%c1] : tensor<16xi16>
%extracted_2 = tensor.extract %source[%c7] : tensor<16xi16>
%inserted_3 = tensor.insert %extracted_2 into %inserted_1[%c2] : tensor<16xi16>
// ... continues for all 16 indices with shift of -5
%extracted_28 = tensor.extract %source[%c4] : tensor<16xi16>
%final = tensor.insert %extracted_28 into %inserted_27[%c15] : tensor<16xi16>
yield %final : tensor<16xi16>
```

**After collapse-insertion-chains:**

```mlir
%result = tensor_ext.rotate %source, %c11 : tensor<16xi16>  // shift = 16-5 = 11
yield %result : tensor<16xi16>
```

## Algorithm Details

### Shift Calculation

For 1D tensors, the shift is calculated as:

```
shift = (extraction_index - insertion_index) mod tensor_size
```

### Chain Traversal

The pass follows the value-semantic chain:

1. Start from first insertion operation
1. Follow users of insertion results
1. Validate consistent extraction source and shift pattern
1. Ensure complete coverage of destination tensor

### Validation Checks

- **Completeness**: All tensor indices must be written
- **Consistency**: All extractions from same source with same shift
- **Constants**: All indices must be compile-time constants

## Prerequisites

For optimal results, run these passes first:

- `--canonicalize`: Folds constant expressions and simplifies indices
- `--sccp`: Propagates constants through control flow if needed

## Limitations

- **1D tensors only**: Currently limited to one-dimensional tensors
- **Complete chains**: Partial chains cannot be optimized
- **Constant indices**: All indices must be constant at compile time
- **Single source**: All extractions must come from the same tensor

## Related Passes

- **Primary use**: Cleanup after `insert-rotate`
- **Prerequisites**: May need `--canonicalize` or `--sccp` for constant folding
- **Synergy**: Works well with `--cse` for removing dead operations
- **Pipeline**: Part of the FHE vectorization cleanup sequence
