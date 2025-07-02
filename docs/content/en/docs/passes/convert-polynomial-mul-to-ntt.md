---
title: Convert Polynomial Multiplication to NTT
weight: 13
---

## Overview

The `convert-polynomial-mul-to-ntt` pass converts polynomial multiplication
operations to their Number Theoretic Transform (NTT) equivalents when possible.
This optimization leverages the fact that polynomial multiplication can be
efficiently computed using NTT, which transforms the O(n²) convolution operation
into O(n log n) complexity.

## Input/Output

- **Input**: MLIR module with polynomial dialect multiplication operations
- **Output**: Same module with polynomial multiplications replaced by NTT-based
  sequences (forward NTT, pointwise multiplication, inverse NTT)

## Options

This pass takes no options - it automatically applies the transformation where
beneficial and mathematically valid.

## Usage Examples

```bash
# Convert polynomial multiplications to NTT form
heir-opt --convert-polynomial-mul-to-ntt input.mlir

# Use in a typical polynomial optimization pipeline
heir-opt --convert-polynomial-mul-to-ntt --canonicalize input.mlir
```

## When to Use

This pass should be used in compilation pipelines that:

1. **Process Polynomial Operations**: When the IR contains polynomial
   multiplication operations
1. **Target NTT-Optimized Backends**: When targeting implementations with
   efficient NTT support
1. **Optimize Performance**: When polynomial multiplication is a performance
   bottleneck

Common use cases:

- FHE scheme implementations (BGV, BFV, CKKS)
- Cryptographic protocol compilation
- Signal processing applications with convolution operations

## Mathematical Background

The transformation is based on the convolution theorem:

- **Direct Multiplication**: `c(x) = a(x) * b(x)` requires O(n²) operations
- **NTT-Based Multiplication**:
  1. `A = NTT(a)` - Forward transform
  1. `B = NTT(b)` - Forward transform
  1. `C = A ⊙ B` - Pointwise multiplication in frequency domain
  1. `c = INTT(C)` - Inverse transform

This reduces complexity from O(n²) to O(n log n).

## Transformation Conditions

The pass applies the transformation when:

- Polynomial coefficients are over a suitable finite field
- Ring size is compatible with NTT (typically powers of 2)
- Modulus supports the required roots of unity
- Backend has efficient NTT implementations

## Example Transformation

**Input:**

```mlir
func.func @poly_multiply(%p1: !polynomial.polynomial<#ring>, %p2: !polynomial.polynomial<#ring>) -> !polynomial.polynomial<#ring> {
  %result = polynomial.mul %p1, %p2 : !polynomial.polynomial<#ring>
  return %result : !polynomial.polynomial<#ring>
}
```

**Output:**

```mlir
func.func @poly_multiply(%p1: !polynomial.polynomial<#ring>, %p2: !polynomial.polynomial<#ring>) -> !polynomial.polynomial<#ring> {
  // Forward NTT on first operand
  %ntt1 = polynomial.ntt %p1 : !polynomial.polynomial<#ring> -> !polynomial.polynomial<#ntt_ring>

  // Forward NTT on second operand
  %ntt2 = polynomial.ntt %p2 : !polynomial.polynomial<#ring> -> !polynomial.polynomial<#ntt_ring>

  // Pointwise multiplication in NTT domain
  %ntt_product = mod_arith.mul %ntt1, %ntt2 : !polynomial.polynomial<#ntt_ring>

  // Inverse NTT to get result
  %result = polynomial.intt %ntt_product : !polynomial.polynomial<#ntt_ring> -> !polynomial.polynomial<#ring>

  return %result : !polynomial.polynomial<#ring>
}
```

## Performance Benefits

- **Asymptotic Improvement**: O(n²) → O(n log n) complexity
- **Cache Efficiency**: NTT operations have better memory access patterns
- **Hardware Acceleration**: Many backends provide optimized NTT implementations
- **Parallelization**: NTT operations can be efficiently parallelized

## Compatibility Requirements

- **Ring Characteristics**: Modulus must support sufficient roots of unity
- **Size Constraints**: Polynomial degree should be power of 2 for optimal NTT
  performance
- **Backend Support**: Target backend must implement NTT operations efficiently

## Integration Notes

This pass works well with:

- **Canonicalization**: Clean up redundant operations after transformation
- **Folding Passes**: Constant fold NTT operations when possible
- **Backend-Specific Optimizations**: Allow backends to further optimize NTT
  sequences
