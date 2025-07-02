---
title: lwe-to-polynomial
weight: 32
---

## Overview

Converts generic LWE dialect operations to polynomial-based representations,
exposing the underlying mathematical structure of FHE operations for analysis
and polynomial-level optimizations. This pass enables mathematical analysis,
symbolic computation, and research into the algebraic foundations of homomorphic
encryption.

## Input/Output

- **Input**: IR with generic LWE dialect operations (`lwe.add`, `lwe.mul`, etc.)
- **Output**: IR with polynomial dialect operations (`polynomial.add`,
  `polynomial.mul`, etc.) representing the algebraic structure

## Options

This pass does not currently expose command-line options.

## Usage Examples

```bash
# Basic usage for polynomial representation
heir-opt --lwe-to-polynomial input.mlir

# Pipeline for mathematical analysis
heir-opt --secret-to-bgv --bgv-to-lwe --lwe-to-polynomial input.mlir

# Combined with polynomial optimizations
heir-opt --lwe-to-polynomial --canonicalize input.mlir
```

## When to Use

- **Mathematical Analysis**: When analyzing the algebraic properties of FHE
  algorithms
- **Research Applications**: For studying polynomial structures in FHE
  operations
- **Educational Purposes**: To understand the mathematical foundations of FHE
- **Algorithm Development**: When developing new FHE algorithms based on
  polynomial analysis
- **Verification**: For mathematical proofs and verification of FHE properties

## Prerequisites

- Input should contain generic LWE dialect operations
- Should run after scheme-to-LWE conversion passes
- Best used for analysis rather than production code generation

## Technical Notes

- **Algebraic Structure**: Preserves the polynomial ring structure underlying
  FHE operations
- **Mathematical Foundation**: Maps ciphertext operations to polynomial
  arithmetic
- **Symbolic Computation**: Enables symbolic analysis of encrypted data
  operations
- **Degree Analysis**: Allows analysis of polynomial degrees and coefficients
- **Research Tool**: Primarily intended for research and educational
  applications

## Analysis Capabilities

- **Degree Tracking**: Monitor polynomial degree growth through operations
- **Coefficient Analysis**: Examine coefficient patterns and properties
- **Algebraic Optimization**: Apply polynomial-level optimization strategies
- **Symbolic Computation**: Perform symbolic analysis on FHE operations
- **Mathematical Verification**: Enable mathematical proofs and verification

## Related Passes

- `bgv-to-lwe`: Provides LWE operations for polynomial conversion (should run
  before)
- `ckks-to-lwe`: Alternative source of LWE operations (should run before)
- `lower-polynomial-eval`: Evaluates polynomial expressions (may run after)
- `polynomial-approximation`: Polynomial approximation optimization (may run
  after)
