---
title: HEIR Transformation Passes
weight: 40
---

## Overview

This section documents HEIR's transformation passes that convert and optimize IR
for homomorphic encryption compilation. The passes are organized by
functionality and typical pipeline placement.

## Layout & Vectorization Passes

These passes optimize data layout and enable SIMD operations for FHE schemes:

- **[layout-propagation](layout-propagation)** - Establishes initial layout
  assignments through forward dataflow analysis
- **[layout-optimization](layout-optimization)** - Optimizes layout conversions
  using greedy cost minimization
- **[straight-line-vectorize](straight-line-vectorize)** - Basic vectorization
  for straight-line programs
- **[insert-rotate](insert-rotate)** - HECO-style SIMD vectorization using
  rotation operations
- **[collapse-insertion-chains](collapse-insertion-chains)** - Cleanup pass
  converting insertion chains to rotations

## Typical Pipeline Usage

For FHE compilation with SIMD optimization:

```bash
heir-opt \
  --layout-propagation \
  --layout-optimization \
  --straight-line-vectorize \
  --insert-rotate \
  --cse --canonicalize \
  --collapse-insertion-chains \
  input.mlir
```

## Pass Categories

### Data Layout Passes

- Control how plaintext data is packed into ciphertexts
- Optimize ciphertext data movement operations
- Enable efficient SIMD computation patterns

### Vectorization Passes

- Convert scalar operations to vector/tensor operations
- Insert rotation operations for FHE SIMD patterns
- Clean up and optimize vectorized code

### Analysis Passes

- Provide dataflow analysis for layout decisions
- Support cost modeling for optimization passes
- Enable reusable transformation infrastructure

## Documentation Structure

Each pass is documented with:

- **Purpose and overview** - What the pass does and why
- **Input/Output specifications** - Expected IR dialects and patterns
- **Command-line options** - Available configuration parameters
- **Usage examples** - Common invocation patterns
- **Pipeline placement** - Where to use in compilation flows
- **Implementation details** - For developers extending passes
