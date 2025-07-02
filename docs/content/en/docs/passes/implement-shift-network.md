---
title: Implement Shift Network
weight: 31
---

## Overview

The implement-shift-network pass converts `tensor_ext.permute` operations into
efficient networks of `tensor_ext.rotate` operations. This pass implements the
Vos-Vos-Erkin algorithm for minimizing permutation latency using graph coloring
techniques. It's particularly useful for FHE schemes where permutations need to
be implemented as sequences of rotation and masking operations.

## Input/Output

- **Input**: IR containing `tensor_ext.permute` operations with either explicit
  permutation maps or affine map representations
- **Output**: Shift networks composed of rotation operations, masked
  multiplications, and additions

## Options

- `--ciphertext-size=<int>`: Power of two length of the ciphertexts the data is
  packed in (default: 1024)

## Usage Examples

```bash
heir-opt --implement-shift-network input.mlir
```

With custom ciphertext size:

```bash
heir-opt --implement-shift-network=ciphertext-size=16 input.mlir
```

## When to Use

Use this pass when you have:

1. Permutation operations that need to be lowered to rotation-based
   implementations
1. FHE schemes that support native rotation operations (BGV, CKKS)
1. IR with `tensor_ext.permute` operations that specify data rearrangement
1. Need for efficient permutation networks with minimized latency

Typical placement in compilation pipelines:

1. After high-level operations have been converted to tensor permutations
1. Before lowering to scheme-specific dialects
1. Often combined with other vectorization passes
1. May be followed by canonicalization to optimize the generated networks

## How It Works

The pass implements the Vos-Vos-Erkin algorithm:

1. **Graph Construction**: Builds a permutation graph from the input
   specification
1. **Graph Coloring**: Groups permutation elements that can be processed in
   parallel
1. **Network Generation**: Creates rotation and masking operations for each
   color group
1. **Optimization**: Minimizes the overall latency of the permutation network

## Example

**Before transformation:**

```mlir
#map = dense<[13, 8, 4, 0, 11, 7, 14, 5, 15, 3, 12, 6, 10, 2, 9, 1]> : tensor<16xi64>
func.func @figure3(%0: tensor<16xi32>) -> tensor<16xi32> {
  %1 = tensor_ext.permute %0 {permutation = #map} : tensor<16xi32>
  return %1 : tensor<16xi32>
}
```

**After transformation:**

```mlir
func.func @figure3(%arg0: tensor<16xi32>) -> tensor<16xi32> {
  %cst = arith.constant dense<[1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]> : tensor<16xi32>
  %0 = arith.muli %arg0, %cst : tensor<16xi32>
  %c1_i32 = arith.constant 1 : i32
  %1 = tensor_ext.rotate %0, %c1_i32 : tensor<16xi32>, i32

  %cst_0 = arith.constant dense<[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<16xi32>
  %2 = arith.muli %arg0, %cst_0 : tensor<16xi32>
  %c2_i32 = arith.constant 2 : i32
  %3 = tensor_ext.rotate %2, %c2_i32 : tensor<16xi32>, i32
  %4 = arith.addi %1, %3 : tensor<16xi32>

  // ... additional groups ...

  return %22 : tensor<16xi32>
}
```

## Algorithm Details

The Vos-Vos-Erkin method:

1. **Partitioning**: Splits the permutation into independent groups
1. **Masking**: Uses plaintext-ciphertext masks to select elements
1. **Rotation**: Applies rotations to move elements to target positions
1. **Combining**: Adds the results from different groups together

## Benefits

- **Minimized Latency**: Uses graph coloring to reduce critical path length
- **Efficient Implementation**: Leverages native FHE rotation operations
- **Parallel Groups**: Independent groups can be processed in parallel
- **Proven Algorithm**: Based on published research with theoretical guarantees

## Limitations

- Ciphertext size must be a power of 2
- Optimized for cyclic permutations (best suited for SIMD packing schemes)
- May generate large intermediate constants for complex permutations
- Performance depends on the permutation pattern complexity

## Research Background

This pass implements the algorithm from:
["Efficient Circuits for Permuting and Mapping Packed Values Across Leveled Homomorphic Ciphertexts"](https://link.springer.com/chapter/10.1007/978-3-031-17140-6_20)
by Vos, Vos, and Erkin (2022).

## Related Passes

- Often used after passes that generate permutation operations
- Works well with `layout-propagation` for SIMD layout management
- Can be combined with `rotate-and-reduce` for comprehensive rotation
  optimization
- May benefit from subsequent canonicalization passes
