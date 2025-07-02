---
title: Lower Polynomial Eval
weight: 17
---

## Overview

The `lower-polynomial-eval` pass converts high-level `polynomial.eval`
operations into sequences of arithmetic operations. This transformation is
crucial for FHE applications where polynomial evaluation must be implemented
using the limited arithmetic operations available in encrypted computation.

## Input/Output

- **Input**: MLIR IR containing `polynomial.eval` operations from the polynomial
  dialect
- **Output**: MLIR IR with arithmetic operation sequences that implement
  polynomial evaluation

## Options

- **`method`**: Selects the polynomial evaluation algorithm:
  - `auto` (default): Automatically selects the best method based on polynomial
    characteristics
  - `horner`: Horner's method for monomial basis polynomials
  - `ps`: Paterson-Stockmeyer method for monomial basis polynomials
  - `pscheb`: Paterson-Stockmeyer method for Chebyshev basis polynomials

## Usage Examples

```bash
# Lower polynomial evaluation with automatic method selection
heir-opt --lower-polynomial-eval input.mlir

# Use Horner's method specifically
heir-opt --lower-polynomial-eval=method=horner input.mlir

# Use Paterson-Stockmeyer method
heir-opt --lower-polynomial-eval=method=ps input.mlir

# Use Chebyshev-basis Paterson-Stockmeyer
heir-opt --lower-polynomial-eval=method=pscheb input.mlir
```

## When to Use

This pass is essential when:

1. **FHE polynomial approximation**: Converting polynomial approximations of
   functions into FHE-compatible arithmetic
1. **Function evaluation**: When implementing mathematical functions through
   polynomial approximations
1. **Performance optimization**: Choosing optimal evaluation algorithms based on
   polynomial characteristics
1. **Backend preparation**: Before lowering to target dialects that don't
   support high-level polynomial operations
1. **Numerical algorithm implementation**: For implementing efficient polynomial
   evaluation in various computational contexts

## Implementation Details

**Evaluation Methods:**

**Horner's Method (`horner`):**

- **Complexity**: O(n) sequential operations for degree-n polynomial
- **Best for**: Low-degree polynomials, sequential evaluation contexts
- **Numerical stability**: Generally good for most polynomials
- **Parallelization**: Limited due to sequential dependency

**Paterson-Stockmeyer (`ps`):**

- **Complexity**: O(√n) parallel depth for degree-n polynomial
- **Best for**: High-degree polynomials, parallel evaluation contexts
- **Trade-off**: Uses more total operations but enables better parallelization
- **FHE advantage**: Reduces circuit depth, which is often more important than
  total operation count

**Automatic Selection (`auto`):**

- Uses heuristics based on polynomial degree and target dialect characteristics
- Considers both computational complexity and parallelization opportunities
- Adapts to the specific requirements of the target arithmetic dialect

**Interface-Based Design:** The pass uses the `DialectPolynomialEvalInterface`
to adapt to different target dialects:

- **Scalar Operations**: Maps polynomial operations to appropriate scalar
  multiplication and addition
- **Constant Materialization**: Handles dialect-specific constant creation
- **Type Adaptation**: Ensures compatibility with target dialect type systems

**Pipeline Integration:**

- Typically follows polynomial approximation passes
- Usually precedes final arithmetic lowering and optimization
- Integrates with FHE-specific optimization pipelines for encrypted computation
