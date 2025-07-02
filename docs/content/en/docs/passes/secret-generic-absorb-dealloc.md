---
title: Secret Generic Absorb Dealloc
weight: 15
---

## Overview

The `secret-generic-absorb-dealloc` pass moves memory deallocation operations
(`memref.dealloc`) into the body of `secret.generic` operations when the
corresponding memory allocations and usage are contained entirely within the
generic body. This improves memory management locality and ensures proper
cleanup within encrypted computation regions.

## Input/Output

- **Input**: IR with `secret.generic` operations that allocate and use memrefs
  internally, with dealloc operations outside the generic
- **Output**: IR with `secret.generic` operations containing the dealloc
  operations for internally managed memrefs

## Options

This pass has no configurable options.

## Usage Examples

```bash
heir-opt --secret-generic-absorb-dealloc input.mlir
```

### Example Input

```mlir
func.func @main(%arg0: !secret.secret<i32>) -> !secret.secret<i32> {
  %0 = secret.generic(%arg0 : !secret.secret<i32>) {
  ^bb0(%arg1: i32):
    %alloc = memref.alloc() : memref<4xi32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    memref.store %arg1, %alloc[%c0] : memref<4xi32>
    memref.store %arg1, %alloc[%c1] : memref<4xi32>
    memref.store %arg1, %alloc[%c2] : memref<4xi32>
    memref.store %arg1, %alloc[%c3] : memref<4xi32>

    %sum = memref.load %alloc[%c0] : memref<4xi32>
    secret.yield %sum : i32
  } -> !secret.secret<i32>

  // Deallocation happens outside the generic
  %alloc_external = memref.alloc() : memref<4xi32>  // This gets deallocated externally
  memref.dealloc %alloc_external : memref<4xi32>

  return %0 : !secret.secret<i32>
}
```

### Example Output

```mlir
func.func @main(%arg0: !secret.secret<i32>) -> !secret.secret<i32> {
  %0 = secret.generic(%arg0 : !secret.secret<i32>) {
  ^bb0(%arg1: i32):
    %alloc = memref.alloc() : memref<4xi32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    memref.store %arg1, %alloc[%c0] : memref<4xi32>
    memref.store %arg1, %alloc[%c1] : memref<4xi32>
    memref.store %arg1, %alloc[%c2] : memref<4xi32>
    memref.store %arg1, %alloc[%c3] : memref<4xi32>

    %sum = memref.load %alloc[%c0] : memref<4xi32>

    // Deallocation now happens inside the generic
    memref.dealloc %alloc : memref<4xi32>

    secret.yield %sum : i32
  } -> !secret.secret<i32>

  return %0 : !secret.secret<i32>
}
```

### Complex Example with Multiple Allocations

**Input:**

```mlir
func.func @process(%arg0: !secret.secret<tensor<8xi32>>) -> !secret.secret<i32> {
  %0 = secret.generic(%arg0 : !secret.secret<tensor<8xi32>>) {
  ^bb0(%arg1: tensor<8xi32>):
    // Convert tensor to memref for processing
    %alloc1 = memref.alloc() : memref<8xi32>
    %alloc2 = memref.alloc() : memref<1xi32>

    // Some computation using the allocated memrefs
    %c0 = arith.constant 0 : index
    %c7 = arith.constant 7 : index
    %first = tensor.extract %arg1[%c0] : tensor<8xi32>
    %last = tensor.extract %arg1[%c7] : tensor<8xi32>
    %sum = arith.addi %first, %last : i32

    memref.store %sum, %alloc2[%c0] : memref<1xi32>
    %result = memref.load %alloc2[%c0] : memref<1xi32>

    secret.yield %result : i32
  } -> !secret.secret<i32>

  // External cleanup would be absorbed
  %external_alloc1 = memref.alloc() : memref<8xi32>
  %external_alloc2 = memref.alloc() : memref<1xi32>
  memref.dealloc %external_alloc1 : memref<8xi32>
  memref.dealloc %external_alloc2 : memref<1xi32>

  return %0 : !secret.secret<i32>
}
```

**Output:**

```mlir
func.func @process(%arg0: !secret.secret<tensor<8xi32>>) -> !secret.secret<i32> {
  %0 = secret.generic(%arg0 : !secret.secret<tensor<8xi32>>) {
  ^bb0(%arg1: tensor<8xi32>):
    %alloc1 = memref.alloc() : memref<8xi32>
    %alloc2 = memref.alloc() : memref<1xi32>

    %c0 = arith.constant 0 : index
    %c7 = arith.constant 7 : index
    %first = tensor.extract %arg1[%c0] : tensor<8xi32>
    %last = tensor.extract %arg1[%c7] : tensor<8xi32>
    %sum = arith.addi %first, %last : i32

    memref.store %sum, %alloc2[%c0] : memref<1xi32>
    %result = memref.load %alloc2[%c0] : memref<1xi32>

    // Deallocation now happens before yielding
    memref.dealloc %alloc1 : memref<8xi32>
    memref.dealloc %alloc2 : memref<1xi32>

    secret.yield %result : i32
  } -> !secret.secret<i32>

  return %0 : !secret.secret<i32>
}
```

## When to Use

The `secret-generic-absorb-dealloc` pass should be used:

1. **After memory allocation transformations** that introduce temporary memrefs
   within generic bodies
1. **To improve memory management locality** by keeping allocation and
   deallocation operations together
1. **Before function extraction** to ensure extracted functions properly manage
   their memory
1. **For memory safety** by ensuring deallocations happen in the right scope
1. **To simplify memory analysis** by localizing memory lifetime management
1. **Before backend lowering** that expects proper memory management within
   computation regions

This pass is particularly important for maintaining proper memory management
semantics when working with temporary memory allocations within encrypted
computation regions.
