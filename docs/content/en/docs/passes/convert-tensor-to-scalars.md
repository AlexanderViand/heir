---
title: Convert Tensor to Scalars
weight: 33
---

## Overview

The convert-tensor-to-scalars pass "unrolls" tensors of static shape into
individual scalar values. This transformation is useful for optimizing small
tensors by avoiding tensor overhead and enabling more aggressive scalar
optimizations. The pass converts tensor operations into sequences of scalar
operations on the individual tensor elements.

## Input/Output

- **Input**: IR with static-shaped tensors and tensor management operations
  (insert, extract, from_elements)
- **Output**: IR where small tensors are replaced with scalar value ranges and
  tensor operations are converted to scalar equivalents

## Options

- `--max-size=<int>`: Limits unrolling to tensors with at most max-size elements
  (default: 8)

## Usage Examples

```bash
heir-opt --convert-tensor-to-scalars input.mlir
```

With custom size limit:

```bash
heir-opt --convert-tensor-to-scalars=max-size=16 input.mlir
```

## When to Use

Use this pass when you have:

1. Small static-shaped tensors that would benefit from scalar treatment
1. IR where tensor overhead outweighs the benefits of tensor operations
1. Code with many tensor.extract/tensor.insert operations
1. Applications where scalar optimization is more effective than tensor
   optimization

Typical placement in compilation pipelines:

1. After elementwise operations have been converted to scalar operations
1. Before scalar optimization passes that can benefit from explicit scalar
   operations
1. Early in the pipeline, before expensive tensor transformations
1. After `--convert-elementwise-to-affine` to prepare suitable input patterns

## How It Works

The pass operates through pattern matching and conversion:

1. **Pattern Recognition**: Identifies tensor operations that can be converted
   to scalars
1. **Size Filtering**: Only processes tensors within the specified size limit
1. **Value Range Creation**: Converts tensors to ValueRange containing
   individual scalars
1. **Operation Conversion**: Replaces tensor operations with scalar equivalents
1. **Folder Application**: Applies folding patterns to simplify the result

## Supported Patterns

Currently supports:

1. **tensor.from_elements**: Converts to scalar inputs directly
1. **tensor.insert**: Updates the scalar ValueRange at the specified index
1. **Folding Patterns**: Simplifies patterns like extract(from_elements)

## Example

**Before conversion:**

```mlir
func.func @example() -> tensor<4xi32> {
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32

  %0 = tensor.from_elements %c1, %c2, %c3, %c4 : tensor<4xi32>
  %c5 = arith.constant 5 : i32
  %c1_idx = arith.constant 1 : index
  %1 = tensor.insert %c5 into %0[%c1_idx] : tensor<4xi32>
  return %1 : tensor<4xi32>
}
```

**After conversion:**

```mlir
func.func @example() -> tensor<4xi32> {
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32
  %c5 = arith.constant 5 : i32

  // Tensor represented as individual scalars: [%c1, %c5, %c3, %c4]
  %result = tensor.from_elements %c1, %c5, %c3, %c4 : tensor<4xi32>
  return %result : tensor<4xi32>
}
```

## Benefits

- **Scalar Optimization**: Enables aggressive scalar-level optimizations
- **Reduced Overhead**: Eliminates tensor management overhead for small tensors
- **Explicit Operations**: Makes data flow explicit for optimization passes
- **Simplified Patterns**: Creates patterns amenable to further optimization

## Limitations

- **Size Restriction**: Only processes tensors up to the maximum size limit
- **Static Shapes**: Requires statically-shaped tensors
- **Limited Operations**: Currently supports only basic tensor management
  operations
- **Code Explosion**: Can significantly increase IR size for larger tensors
- **Compilation Time**: May be slow for large tensors (hence the size limit)

## Design Considerations

The pass is designed to work on IR where:

- Elementwise operations have already been converted to scalar operations
- Only tensor "management" operations (insert/extract/from_elements) remain
- Tensor sizes are small enough that scalar treatment is beneficial

## Future Enhancements

TODO items include:

- Support for `tensor.slice` operations (#1023)
- Extended support for more tensor operations
- Better heuristics for when scalar conversion is beneficial

## Related Passes

- Use after `--convert-elementwise-to-affine` to prepare suitable input
- Works well with scalar optimization passes
- Can be combined with constant folding passes
- Often followed by canonicalization to clean up the result
