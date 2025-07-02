---
title: Shape Inference
weight: 35
---

## Overview

The shape-inference pass infers the shapes of shaped types in functions,
starting from function arguments annotated with shape information. This pass is
particularly useful when working with the Python frontend, which can infer
tensor ranks but not the specific dimensions. The pass propagates shape
information through operations that implement the InferTypeOpInterface.

## Input/Output

- **Input**: IR with function arguments annotated with `{shape.shape}`
  attributes and operations with partially unknown shapes
- **Output**: IR with complete shape information propagated throughout the
  function

## Options

This pass has no command-line options.

## Usage Examples

```bash
heir-opt --shape-inference input.mlir
```

Typically used in frontend compilation pipelines:

```bash
heir-opt --shape-inference --canonicalize input.mlir
```

## When to Use

Use this pass when you have:

1. IR generated from Python frontend with incomplete shape information
1. Functions with shaped types that need complete dimension inference
1. Operations implementing InferTypeOpInterface that can propagate shapes
1. Need to resolve tensor dimensions before further optimization

Typical placement in compilation pipelines:

1. Early in the pipeline, immediately after frontend IR generation
1. Before passes that require complete shape information
1. After type conversion but before shape-dependent optimizations
1. Before vectorization or layout passes that need precise tensor dimensions

## How It Works

The pass performs shape propagation through dataflow analysis:

1. **Seed Identification**: Starts from function arguments with `{shape.shape}`
   attributes
1. **Interface Usage**: Leverages InferTypeOpInterface for shape propagation
1. **Forward Propagation**: Propagates shape information through the computation
   graph
1. **Type Updates**: Updates types with inferred shape information

## Example

**Before shape inference:**

```mlir
func.func @example(%arg0: tensor<?x?xf32> {shape.shape = [64, 32]}) -> tensor<?xf32> {
  %0 = "some.op"(%arg0) : (tensor<?x?xf32>) -> tensor<?xf32>
  %1 = "another.op"(%0) : (tensor<?xf32>) -> tensor<?xf32>
  return %1 : tensor<?xf32>
}
```

**After shape inference:**

```mlir
func.func @example(%arg0: tensor<64x32xf32>) -> tensor<64xf32> {
  %0 = "some.op"(%arg0) : (tensor<64x32xf32>) -> tensor<64xf32>
  %1 = "another.op"(%0) : (tensor<64xf32>) -> tensor<64xf32>
  return %1 : tensor<64xf32>
}
```

## Shape Annotation Format

Function arguments should be annotated with shape information:

```mlir
func.func @example(
  %arg0: tensor<?x?xf32> {shape.shape = [128, 64]},
  %arg1: tensor<?xf32> {shape.shape = [64]}
) -> tensor<?xf32>
```

## Supported Operations

The pass works with any operation that implements:

- **InferTypeOpInterface**: For automatic shape inference
- Standard MLIR operations with well-defined shape semantics
- Custom operations that provide shape inference capabilities

## Benefits

- **Complete Shape Information**: Resolves unknown dimensions throughout the IR
- **Frontend Integration**: Essential for Python frontend compilation
- **Optimization Enablement**: Enables shape-dependent optimizations
- **Type Safety**: Provides complete type information for validation

## Use Cases

### Python Frontend Integration

```python
# Python function generates IR with partial shape info
def matmul(a, b):  # Shapes inferred at runtime
    return a @ b

# After shape-inference, complete shapes are available
```

### Dynamic to Static Conversion

```mlir
// Before: Dynamic shapes from frontend
tensor<?x?xf32> -> tensor<?xf32>

// After: Static shapes for optimization
tensor<64x32xf32> -> tensor<64xf32>
```

## Limitations

- Only works with operations implementing InferTypeOpInterface
- Requires initial shape annotations on function arguments
- Cannot infer shapes for operations without proper interface implementation
- Limited to shaped types (tensors, memrefs)

## Integration with Frontend

This pass is designed to work with the HEIR Python frontend:

1. **Python Analysis**: Frontend infers ranks but not dimensions
1. **Shape Annotations**: Runtime shape information is preserved as attributes
1. **IR Generation**: Initial IR has dynamic shapes with shape hints
1. **Shape Inference**: This pass resolves the dynamic shapes to static ones

## Related Passes

- Essential for Python frontend workflows
- Often used before vectorization passes
- Works well with type conversion passes
- May be followed by canonicalization to clean up types
- Enables layout propagation and other shape-dependent optimizations
