---
title: Tensor Linalg to Affine Loops
weight: 37
---

## Overview

The tensor-linalg-to-affine-loops pass is a specialized version of the standard
`convert-linalg-to-affine-loops` pass that handles loops with tensor semantics.
This pass is primarily designed to support the conversion of `linalg.generic`
operations that implement `tensor_ext.assign_layout` operations, enabling proper
lowering of layout assignment operations to affine loop structures.

## Input/Output

- **Input**: IR containing `linalg.generic` operations with tensor semantics,
  particularly those implementing `tensor_ext.assign_layout`
- **Output**: IR where linalg operations are converted to affine loop nests that
  preserve tensor semantics

## Options

This pass has no command-line options.

## Usage Examples

```bash
heir-opt --tensor-linalg-to-affine-loops input.mlir
```

Typically used in layout assignment pipelines:

```bash
heir-opt --layout-propagation --tensor-linalg-to-affine-loops --canonicalize input.mlir
```

## When to Use

Use this pass when you have:

1. `linalg.generic` operations that implement `tensor_ext.assign_layout`
   functionality
1. Layout assignment operations that need to be lowered to explicit loops
1. IR where tensor layout operations need to be converted to affine structures
1. Compilation pipelines that benefit from explicit loop representation of
   tensor operations

Typical placement in compilation pipelines:

1. After layout propagation passes that generate `tensor_ext.assign_layout`
   operations
1. Before further affine optimizations that operate on loop structures
1. When explicit loop representation of tensor operations is needed
1. Often followed by affine optimization passes

## How It Works

The pass extends the standard linalg-to-affine-loops conversion with:

1. **Tensor Semantics Preservation**: Maintains tensor operation semantics
   during conversion
1. **Layout Assignment Support**: Properly handles `tensor_ext.assign_layout`
   operations
1. **Affine Loop Generation**: Creates affine loop nests that implement the
   tensor operations
1. **Semantic Mapping**: Ensures the affine loops correctly implement the
   original tensor semantics

## Example

**Before conversion:**

```mlir
func.func @assign_layout(%input: tensor<32x32xi16>) -> tensor<1024xi16> {
  %0 = tensor_ext.assign_layout %input {
    tensor_ext.layout = #row_major_layout
  } : tensor<32x32xi16> -> tensor<1024xi16>
  return %0 : tensor<1024xi16>
}
```

**After conversion (conceptual):**

```mlir
func.func @assign_layout(%input: tensor<32x32xi16>) -> tensor<1024xi16> {
  %output = tensor.empty() : tensor<1024xi16>
  %result = affine.for %i = 0 to 32 {
    %partial = affine.for %j = 0 to 32 iter_args(%iter = %output) -> tensor<1024xi16> {
      %elem = tensor.extract %input[%i, %j] : tensor<32x32xi16>
      %flat_idx = affine.apply affine_map<(d0, d1) -> (d0 * 32 + d1)>(%i, %j)
      %updated = tensor.insert %elem into %iter[%flat_idx] : tensor<1024xi16>
      affine.yield %updated : tensor<1024xi16>
    }
    affine.yield %partial : tensor<1024xi16>
  }
  return %result : tensor<1024xi16>
}
```

## Relationship to Standard Pass

This pass is a "port" of the standard MLIR `convert-linalg-to-affine-loops` pass
with specific extensions:

- **Tensor Focus**: Specialized for tensor-semantic operations
- **Layout Support**: Enhanced support for layout assignment operations
- **HEIR Integration**: Designed to work with HEIR's tensor_ext dialect

## Use Cases

### Layout Assignment Lowering

Converting high-level layout assignments to explicit loops:

```mlir
// High-level layout assignment
%packed = tensor_ext.assign_layout %cleartext {layout = #simd_layout}

// Converted to explicit affine loops that implement the packing
```

### Tensor Reshaping Operations

Lowering tensor reshaping operations to loop-based implementations:

```mlir
// Before: implicit tensor reshaping
%reshaped = tensor.reshape %input

// After: explicit loops implementing the reshape
```

## Benefits

- **Explicit Loops**: Makes tensor operations explicit as affine loop structures
- **Layout Integration**: Properly handles HEIR's layout assignment operations
- **Optimization Enablement**: Enables affine-based optimizations on tensor
  operations
- **Semantic Preservation**: Maintains the correct semantics of tensor
  operations

## Limitations

- Primarily focused on `tensor_ext.assign_layout` operations
- May not handle all types of linalg operations
- Requires subsequent optimization to clean up generated loops
- Limited to operations that can be represented as affine loops

## Integration with HEIR

This pass is specifically designed for HEIR's compilation pipeline:

1. **tensor_ext Dialect**: Works with HEIR's tensor extension dialect
1. **Layout System**: Integrates with HEIR's layout propagation system
1. **FHE Compilation**: Supports FHE-specific tensor layout requirements

## Related Passes

- Use after `layout-propagation` which generates assign_layout operations
- Often followed by affine optimization passes
- Works well with canonicalization to clean up generated loops
- Can be combined with other affine-to-scalar lowering passes
