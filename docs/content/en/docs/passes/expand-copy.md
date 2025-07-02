---
title: Expand Copy
weight: 23
---

## Overview

The `expand-copy` pass removes `memref.copy` operations by expanding them into
explicit affine loads and stores. This transformation converts high-level memory
copy operations into fine-grained memory operations that can be further
optimized and eventually removed from FHE compilation pipelines.

## Input/Output

- **Input**: IR with `memref.copy` operations
- **Output**: IR with `affine.for` loops (or unrolled loads/stores) containing
  explicit `affine.load` and `affine.store` operations

## Options

- `--disable-affine-loop`: Use unrolled loads and stores instead of affine loops
  (default: false)

## Usage Examples

With affine loops (default):

```bash
heir-opt --expand-copy input.mlir
```

With unrolled loads and stores:

```bash
heir-opt --expand-copy="disable-affine-loop=true" input.mlir
```

### Example Input

```mlir
module {
  func.func @memref_copy() {
    %alloc = memref.alloc() : memref<2x3xi32>
    %alloc_0 = memref.alloc() : memref<2x3xi32>
    memref.copy %alloc, %alloc_0 : memref<2x3xi32> to memref<2x3xi32>
  }
}
```

### Example Output (Default - With Affine Loops)

```mlir
module {
  func.func @memref_copy() {
    %alloc = memref.alloc() : memref<2x3xi32>
    %alloc_0 = memref.alloc() : memref<2x3xi32>
    affine.for %arg0 = 0 to 2 {
      affine.for %arg1 = 0 to 3 {
        %0 = affine.load %alloc[%arg0, %arg1] : memref<2x3xi32>
        affine.store %0, %alloc_0[%arg0, %arg1] : memref<2x3xi32>
      }
    }
  }
}
```

### Example Output (With disable-affine-loop=true)

```mlir
module {
  func.func @memref_copy() {
    %alloc = memref.alloc() : memref<2x3xi32>
    %alloc_0 = memref.alloc() : memref<2x3xi32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    // Unrolled copy operations
    %0 = affine.load %alloc[%c0, %c0] : memref<2x3xi32>
    affine.store %0, %alloc_0[%c0, %c0] : memref<2x3xi32>
    %1 = affine.load %alloc[%c0, %c1] : memref<2x3xi32>
    affine.store %1, %alloc_0[%c0, %c1] : memref<2x3xi32>
    %2 = affine.load %alloc[%c0, %c2] : memref<2x3xi32>
    affine.store %2, %alloc_0[%c0, %c2] : memref<2x3xi32>
    %3 = affine.load %alloc[%c1, %c0] : memref<2x3xi32>
    affine.store %3, %alloc_0[%c1, %c0] : memref<2x3xi32>
    %4 = affine.load %alloc[%c1, %c1] : memref<2x3xi32>
    affine.store %4, %alloc_0[%c1, %c1] : memref<2x3xi32>
    %5 = affine.load %alloc[%c1, %c2] : memref<2x3xi32>
    affine.store %5, %alloc_0[%c1, %c2] : memref<2x3xi32>
  }
}
```

### Complex Example with Multiple Copies

**Input:**

```mlir
func.func @multiple_copies() {
  %input = memref.alloc() : memref<4x4xi32>
  %temp1 = memref.alloc() : memref<4x4xi32>
  %temp2 = memref.alloc() : memref<4x4xi32>
  %output = memref.alloc() : memref<4x4xi32>

  // Chain of copy operations
  memref.copy %input, %temp1 : memref<4x4xi32> to memref<4x4xi32>
  memref.copy %temp1, %temp2 : memref<4x4xi32> to memref<4x4xi32>
  memref.copy %temp2, %output : memref<4x4xi32> to memref<4x4xi32>
}
```

**Output (with disable-affine-loop=false):**

```mlir
func.func @multiple_copies() {
  %input = memref.alloc() : memref<4x4xi32>
  %temp1 = memref.alloc() : memref<4x4xi32>
  %temp2 = memref.alloc() : memref<4x4xi32>
  %output = memref.alloc() : memref<4x4xi32>

  // First copy: input -> temp1
  affine.for %arg0 = 0 to 4 {
    affine.for %arg1 = 0 to 4 {
      %0 = affine.load %input[%arg0, %arg1] : memref<4x4xi32>
      affine.store %0, %temp1[%arg0, %arg1] : memref<4x4xi32>
    }
  }

  // Second copy: temp1 -> temp2
  affine.for %arg0 = 0 to 4 {
    affine.for %arg1 = 0 to 4 {
      %1 = affine.load %temp1[%arg0, %arg1] : memref<4x4xi32>
      affine.store %1, %temp2[%arg0, %arg1] : memref<4x4xi32>
    }
  }

  // Third copy: temp2 -> output
  affine.for %arg0 = 0 to 4 {
    affine.for %arg1 = 0 to 4 {
      %2 = affine.load %temp2[%arg0, %arg1] : memref<4x4xi32>
      affine.store %2, %output[%arg0, %arg1] : memref<4x4xi32>
    }
  }
}
```

### 1D Memref Example

**Input:**

```mlir
func.func @copy_1d() {
  %src = memref.alloc() : memref<8xi64>
  %dst = memref.alloc() : memref<8xi64>
  memref.copy %src, %dst : memref<8xi64> to memref<8xi64>
}
```

**Output (default):**

```mlir
func.func @copy_1d() {
  %src = memref.alloc() : memref<8xi64>
  %dst = memref.alloc() : memref<8xi64>
  affine.for %arg0 = 0 to 8 {
    %0 = affine.load %src[%arg0] : memref<8xi64>
    affine.store %0, %dst[%arg0] : memref<8xi64>
  }
}
```

**Output (with disable-affine-loop=true):**

```mlir
func.func @copy_1d() {
  %src = memref.alloc() : memref<8xi64>
  %dst = memref.alloc() : memref<8xi64>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index

  %0 = affine.load %src[%c0] : memref<8xi64>
  affine.store %0, %dst[%c0] : memref<8xi64>
  %1 = affine.load %src[%c1] : memref<8xi64>
  affine.store %1, %dst[%c1] : memref<8xi64>
  %2 = affine.load %src[%c2] : memref<8xi64>
  affine.store %2, %dst[%c2] : memref<8xi64>
  %3 = affine.load %src[%c3] : memref<8xi64>
  affine.store %3, %dst[%c3] : memref<8xi64>
  %4 = affine.load %src[%c4] : memref<8xi64>
  affine.store %4, %dst[%c4] : memref<8xi64>
  %5 = affine.load %src[%c5] : memref<8xi64>
  affine.store %5, %dst[%c5] : memref<8xi64>
  %6 = affine.load %src[%c6] : memref<8xi64>
  affine.store %6, %dst[%c6] : memref<8xi64>
  %7 = affine.load %src[%c7] : memref<8xi64>
  affine.store %7, %dst[%c7] : memref<8xi64>
}
```

## Pipeline Integration

This pass should be used early in the pipeline:

1. **Before affine loop unrolling** - If using default mode, subsequent passes
   can unroll the generated loops
1. **After tensor-to-memref lowering** - When high-level copy operations have
   been introduced
1. **Before store-to-load forwarding** - To expose load/store patterns for
   optimization

## When to Use

The `expand-copy` pass should be used:

1. **Early in FHE compilation pipelines** to break down high-level memory
   operations
1. **Before affine loop unrolling** when you want the generated loops to be
   unrolled later
1. **With `disable-affine-loop=true`** when you want immediate unrolling and
   don't want loops
1. **After tensor/linalg to memref lowering** that introduces copy operations
1. **Before `unroll-and-forward`** to expose memory access patterns
1. **In ML model pipelines** where tensor operations generate memory copies

This pass is particularly important for FHE compilation as it converts
high-level memory copy semantics into explicit element-wise operations that can
be further optimized and eventually eliminated through store-to-load forwarding.
