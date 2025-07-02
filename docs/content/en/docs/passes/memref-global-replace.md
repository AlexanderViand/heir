---
title: Memref Global Replace
weight: 22
---

## Overview

The `memref-global-replace` pass forwards constant global MemRef values to
referencing affine loads, replacing memory accesses with direct arithmetic
constants. This transformation is essential for FHE compilation as it removes
memory allocations from models (such as TensorFlow model weights) that cannot be
handled in encrypted computation.

## Input/Output

- **Input**: IR with `memref.global` constants and `affine.load` operations
  accessing them with constant indices
- **Output**: IR with direct `arith.constant` values replacing the memory
  accesses, and global memrefs removed

## Options

This pass has no configurable options.

## Usage Examples

```bash
heir-opt --memref-global-replace input.mlir
```

### Example Input

```mlir
module {
  memref.global "private" constant @__constant_8xi16 : memref<2x4xi16> = dense<[[-10, 20, 3, 4], [5, 6, 7, 8]]>

  func.func @main() -> i16 {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = memref.get_global @__constant_8xi16 : memref<2x4xi16>
    %1 = affine.load %0[%c1, %c1 + %c2] : memref<2x4xi16>
    return %1 : i16
  }
}
```

### Example Output

```mlir
module {
  func.func @main() -> i16 {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8_i16 = arith.constant 8 : i16
    return %c8_i16 : i16
  }
}
```

### Complex Example with Multiple Accesses

**Input:**

```mlir
module {
  memref.global "private" constant @weights : memref<3x3xi32> = dense<[[1, 2, 3], [4, 5, 6], [7, 8, 9]]>
  memref.global "private" constant @bias : memref<3xi32> = dense<[10, 20, 30]>

  func.func @linear_layer(%input: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    // Access weight matrix
    %weights_ref = memref.get_global @weights : memref<3x3xi32>
    %w00 = affine.load %weights_ref[%c0, %c0] : memref<3x3xi32>
    %w01 = affine.load %weights_ref[%c0, %c1] : memref<3x3xi32>
    %w02 = affine.load %weights_ref[%c0, %c2] : memref<3x3xi32>

    // Access bias vector
    %bias_ref = memref.get_global @bias : memref<3xi32>
    %b0 = affine.load %bias_ref[%c0] : memref<3xi32>

    // Simple computation
    %mul1 = arith.muli %input, %w00 : i32
    %mul2 = arith.muli %input, %w01 : i32
    %add1 = arith.addi %mul1, %mul2 : i32
    %result = arith.addi %add1, %b0 : i32

    return %result : i32
  }
}
```

**Output:**

```mlir
module {
  func.func @linear_layer(%input: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    // Constants directly replaced from global memrefs
    %c1_i32 = arith.constant 1 : i32   // weights[0][0]
    %c2_i32 = arith.constant 2 : i32   // weights[0][1]
    %c3_i32 = arith.constant 3 : i32   // weights[0][2]
    %c10_i32 = arith.constant 10 : i32 // bias[0]

    // Computation using the constants
    %mul1 = arith.muli %input, %c1_i32 : i32
    %mul2 = arith.muli %input, %c2_i32 : i32
    %add1 = arith.addi %mul1, %mul2 : i32
    %result = arith.addi %add1, %c10_i32 : i32

    return %result : i32
  }
}
```

### Example with Computed Indices

**Input:**

```mlir
module {
  memref.global "private" constant @lookup_table : memref<4xi32> = dense<[100, 200, 300, 400]>

  func.func @lookup(%index: index) -> i32 {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %table = memref.get_global @lookup_table : memref<4xi32>

    // Constant computed index: 1 + 2 = 3
    %computed_idx = arith.addi %c1, %c2 : index
    %value = affine.load %table[%computed_idx] : memref<4xi32>

    return %value : i32
  }
}
```

**Output:**

```mlir
module {
  func.func @lookup(%index: index) -> i32 {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    %computed_idx = arith.addi %c1, %c2 : index
    %c400_i32 = arith.constant 400 : i32  // lookup_table[3]

    return %c400_i32 : i32
  }
}
```

## Requirements and Limitations

1. **Constant Globals**: MemRef globals must be initialized with constant values
   (e.g., `dense<...>`)
1. **Constant Indices**: Affine load access indices must be statically inferable
   constants or affine expressions with constant operands
1. **Loop Unrolling**: Affine loops should be unrolled prior to running this
   pass to expose constant indices
1. **Single Use**: Works best when global memrefs are only accessed for reading

## FHE Compilation Context

This pass is crucial for FHE compilation because:

1. **Memory Elimination**: FHE schemes cannot handle dynamic memory allocations
1. **Model Weights**: TensorFlow and other ML models store weights in global
   memrefs
1. **Constant Propagation**: Enables further optimizations by exposing constants
1. **Compilation Requirements**: Many FHE backends require purely arithmetic
   computations

## When to Use

The `memref-global-replace` pass should be used:

1. **After tensor-to-memref lowering** when ML models have been converted to
   memref operations
1. **After affine loop unrolling** to expose constant access patterns
1. **Before FHE backend lowering** to eliminate memory operations
1. **In ML model compilation pipelines** to handle model weights and parameters
1. **Before secretization** to ensure only arithmetic operations remain
1. **As preparation for `unroll-and-forward`** to handle remaining memory
   operations

This pass is essential in FHE compilation pipelines for ML models, where it
transforms memory-based parameter access into direct constant usage, enabling
the computation to proceed without memory allocations.
