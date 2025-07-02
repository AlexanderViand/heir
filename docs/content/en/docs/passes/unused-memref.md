---
title: Remove Unused MemRef
weight: 38
---

## Overview

The remove-unused-memref pass is a cleanup pass that scans the IR for locally
allocated memrefs that are never used and removes them. This pass helps clean up
IR after other transformations that may have made certain memory allocations
unnecessary, such as store-to-load forwarding optimizations.

## Input/Output

- **Input**: IR with potentially unused memref allocations
- **Output**: Cleaned IR with unused memref allocations removed

## Options

This pass has no command-line options.

## Usage Examples

```bash
heir-opt --remove-unused-memref input.mlir
```

Typically used as a cleanup pass after other optimizations:

```bash
heir-opt --forward-store-to-load --remove-unused-memref input.mlir
```

## When to Use

Use this pass when you have:

1. IR with potentially unused memref allocations
1. After store-to-load forwarding optimizations that eliminate memref usage
1. After other IR transformations that may make allocations unnecessary
1. As a cleanup pass to reduce memory allocation overhead
1. In compilation pipelines where memory allocation minimization is important

Typical placement in compilation pipelines:

1. After optimization passes that eliminate memref usage
1. As a cleanup pass late in the compilation pipeline
1. After `forward-store-to-load` and similar memory optimization passes
1. Before final code generation to minimize unnecessary allocations

## How It Works

The pass performs dead code elimination for memory allocations:

1. **Allocation Detection**: Identifies locally allocated memrefs (e.g.,
   `memref.alloca`)
1. **Usage Analysis**: Analyzes whether each allocation is actually used
1. **Dead Code Elimination**: Removes allocations that have no uses
1. **Scope Limitation**: Only removes locally allocated memrefs, not function
   arguments

## Example

**Before cleanup:**

```mlir
func.func @example() -> i32 {
  %alloc1 = memref.alloca() : memref<16xi32>  // Used
  %alloc2 = memref.alloca() : memref<8xi32>   // Unused - will be removed
  %alloc3 = memref.alloca() : memref<4xi32>   // Unused - will be removed

  %c0 = arith.constant 0 : index
  %c5 = arith.constant 5 : i32

  memref.store %c5, %alloc1[%c0] : memref<16xi32>
  %result = memref.load %alloc1[%c0] : memref<16xi32>

  return %result : i32
}
```

**After cleanup:**

```mlir
func.func @example() -> i32 {
  %alloc1 = memref.alloca() : memref<16xi32>  // Kept - actually used

  %c0 = arith.constant 0 : index
  %c5 = arith.constant 5 : i32

  memref.store %c5, %alloc1[%c0] : memref<16xi32>
  %result = memref.load %alloc1[%c0] : memref<16xi32>

  return %result : i32
}
```

## What Gets Removed

The pass removes:

- `memref.alloca` operations with no uses
- `memref.malloc` operations with no uses
- Other local memory allocation operations that are never referenced

## What Doesn't Get Removed

The pass preserves:

- Function arguments (even if unused)
- Global memrefs
- Memrefs with any usage, even if that usage is later optimized away
- Memrefs that escape the function scope

## Benefits

- **Memory Optimization**: Reduces unnecessary memory allocations
- **Code Cleanup**: Removes dead code that clutters the IR
- **Compilation Efficiency**: Reduces the burden on later compilation phases
- **Resource Management**: Helps minimize memory footprint of generated code

## Use Cases

### After Store-to-Load Forwarding

When store-to-load forwarding eliminates the need for temporary storage:

```mlir
// Before forwarding: memref is used
%temp = memref.alloca() : memref<1xi32>
memref.store %value, %temp[%c0]
%result = memref.load %temp[%c0]

// After forwarding: memref becomes unused and can be removed
%result = %value  // Direct forwarding, %temp is now unused
```

### After Loop Optimizations

When loop optimizations eliminate the need for certain buffers:

```mlir
// Before optimization: buffer used in loop
%buffer = memref.alloca() : memref<10xi32>
// ... complex loop that gets optimized away ...

// After optimization: buffer no longer needed
// Direct computation without intermediate buffer
```

## Pass Scope

This pass operates at the **function level** (`func::FuncOp`), meaning:

- It analyzes each function independently
- It doesn't remove memrefs that are used across function boundaries
- It focuses on locally allocated, function-scoped memory

## Limitations

- Only removes locally allocated memrefs
- Cannot remove memrefs that have any usage, even if that usage is semantically
  dead
- Does not perform interprocedural analysis
- Limited to simple usage analysis (doesn't handle complex aliasing scenarios)

## Related Passes

- Commonly used after `forward-store-to-load` which can create unused
  allocations
- Works well with other memory optimization passes
- Can be combined with dead code elimination passes
- Often used before final memory lowering passes
