---
title: Convert Elementwise to Affine
weight: 14
---

## Overview

The `convert-elementwise-to-affine` pass converts high-level elementwise tensor
operations into explicit affine loop nests that operate on scalar values. This
transformation is essential for lowering tensor abstractions to the scalar
operations required by FHE schemes.

## Input/Output

- **Input**: MLIR IR containing ElementwiseMappable operations on tensors
- **Output**: MLIR IR with affine loops that apply operations to individual
  tensor elements

## Options

- **`convert-ops`**: Comma-separated list of specific operations to convert
  (e.g., `arith.addf,arith.mulf`)
- **`convert-dialects`**: Comma-separated list of dialects to convert all
  operations from (e.g., `arith,bgv`)

## Usage Examples

```bash
# Convert all elementwise operations
heir-opt --convert-elementwise-to-affine input.mlir

# Convert only specific arithmetic operations
heir-opt --convert-elementwise-to-affine=convert-ops=arith.addf,arith.mulf input.mlir

# Convert all operations from specific dialects
heir-opt --convert-elementwise-to-affine=convert-dialects=arith,bgv input.mlir

# Mixed specification
heir-opt --convert-elementwise-to-affine=convert-ops=arith.addf,arith.divf,convert-dialects=bgv input.mlir
```

## When to Use

This pass is essential when:

1. **FHE lowering preparation**: Before converting to FHE schemes that operate
   on individual ciphertext elements
1. **Scalar optimization enablement**: To expose scalar operations for
   subsequent optimization passes
1. **Loop-level optimization**: When you need explicit loops for vectorization
   or parallelization analysis
1. **Selective lowering**: When only certain operations need to be converted
   while preserving high-level abstractions for others
1. **Custom operation handling**: When working with domain-specific operations
   that implement ElementwiseMappable

The pass provides fine-grained control over which operations are lowered, making
it suitable for gradual lowering strategies.

## Implementation Details

The transformation process:

1. **Operation Analysis**: Identifies operations that implement the
   ElementwiseMappable interface
1. **Filtering**: Applies convert-ops and convert-dialects filters to select
   operations for conversion
1. **Loop Generation**: Creates affine.for loops that iterate over tensor
   dimensions
1. **Scalar Operation**: Extracts tensor elements, applies the operation, and
   inserts results
1. **Type Preservation**: Maintains tensor types and shapes throughout the
   transformation

**Selection Criteria:**

- **ElementwiseMappable Interface**: Only operations implementing this interface
  are eligible
- **Operation Filter**: `convert-ops` allows precise control over which
  operations to convert
- **Dialect Filter**: `convert-dialects` enables conversion of all operations
  from specified dialects
- **Combined Filtering**: Both filters can be used together for maximum
  flexibility

**Generated Loop Structure:**

```mlir
// Input: Tensor addition
%result = arith.addf %tensor1, %tensor2 : tensor<4x4xf32>

// Output: Affine loops with scalar operations
%result = affine.for %i = 0 to 4 iter_args(%iter1 = %init) -> tensor<4x4xf32> {
  %inner = affine.for %j = 0 to 4 iter_args(%iter2 = %iter1) -> tensor<4x4xf32> {
    %elem1 = tensor.extract %tensor1[%i, %j] : tensor<4x4xf32>
    %elem2 = tensor.extract %tensor2[%i, %j] : tensor<4x4xf32>
    %sum = arith.addf %elem1, %elem2 : f32
    %new_tensor = tensor.insert %sum into %iter2[%i, %j] : tensor<4x4xf32>
    affine.yield %new_tensor : tensor<4x4xf32>
  }
  affine.yield %inner : tensor<4x4xf32>
}
```

**Benefits:**

- Enables scalar-level optimization and analysis
- Provides explicit loop structure for further transformations
- Maintains parallelization opportunities through affine loops
- Allows selective lowering for incremental compilation strategies
