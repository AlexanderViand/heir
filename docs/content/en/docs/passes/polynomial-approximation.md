---
title: Polynomial Approximation
weight: 36
---

## Overview

The polynomial-approximation pass replaces mathematical operations that are
incompatible with the FHE computational model with polynomial approximations.
Since FHE schemes typically only support addition and multiplication, functions
like exponentials, trigonometric operations, and other transcendental functions
must be approximated using polynomials that can be evaluated using only
FHE-compatible operations.

## Input/Output

- **Input**: IR containing `math` dialect operations and certain `arith` dialect
  operations that need polynomial approximation
- **Output**: IR where incompatible operations are replaced with
  `polynomial.eval` operations containing static polynomial attributes

## Options

This pass has no command-line options. Approximation parameters (degree, domain)
are specified as attributes on individual operations.

## Usage Examples

```bash
heir-opt --polynomial-approximation input.mlir
```

Typically used in FHE compilation pipelines:

```bash
heir-opt --polynomial-approximation --canonicalize input.mlir
```

## When to Use

Use this pass when you have:

1. Mathematical operations incompatible with FHE computation models
1. Transcendental functions that need polynomial approximation
1. IR targeting FHE backends that only support addition and multiplication
1. Applications requiring approximate computation of complex mathematical
   functions

Typical placement in compilation pipelines:

1. Early in FHE compilation pipelines, before lowering to FHE dialects
1. After frontend code generation but before FHE-specific transformations
1. Before passes that require only FHE-compatible operations
1. Often followed by polynomial evaluation optimization passes

## Supported Operations

### Math Dialect Operations

**Unary Functions:**

- `absf` - Absolute value
- `acos` - Inverse cosine
- `acosh` - Inverse hyperbolic cosine
- `asin` - Inverse sine
- `asinh` - Inverse hyperbolic sine
- `atan` - Inverse tangent
- `atanh` - Inverse hyperbolic tangent
- `cbrt` - Cube root
- `ceil` - Ceiling function
- `cos` - Cosine
- `cosh` - Hyperbolic cosine
- `erf` - Error function
- `erfc` - Complementary error function
- `exp` - Exponential
- `exp2` - Base-2 exponential
- `expm1` - exp(x) - 1
- `floor` - Floor function
- `log` - Natural logarithm
- `log10` - Base-10 logarithm
- `log1p` - log(1 + x)
- `log2` - Base-2 logarithm
- `round` - Round to nearest integer
- `roundeven` - Round to nearest even integer
- `rsqrt` - Reciprocal square root
- `sin` - Sine
- `sinh` - Hyperbolic sine
- `sqrt` - Square root
- `tan` - Tangent
- `tanh` - Hyperbolic tangent
- `trunc` - Truncate to integer

**Binary Functions:**

- `atan2` - Two-argument arctangent
- `copysign` - Copy sign operation
- `fpowi` - Floating-point power with integer exponent
- `powf` - Floating-point power

### Arith Dialect Operations

- `maxf` - Maximum of two floats
- `maxnumf` - Numeric maximum (handles NaN)
- `minf` - Minimum of two floats
- `minnumf` - Numeric minimum (handles NaN)

## Example

**Before approximation:**

```mlir
func.func @example(%x: f32) -> f32 {
  %0 = math.exp %x {
    degree = 3 : i32,
    domain_lower = -1.0 : f64,
    domain_upper = 1.0 : f64
  } : f32
  return %0 : f32
}
```

**After approximation:**

```mlir
#ring_f64_ = #polynomial.ring<coefficientType = f64>
!poly = !polynomial.polynomial<ring = #ring_f64_>

func.func @example(%x: f32) -> f32 {
  %0 = polynomial.eval
    #polynomial<typed_float_polynomial <
      0.99458116404270657
    + 0.99565537253615788x
    + 0.54297028147256321x**2
    + 0.17954582110873779x**3> : !poly>, %x : f32
  return %0 : f32
}
```

## Approximation Parameters

Operations can be annotated with approximation parameters:

- `degree`: Polynomial degree for approximation
- `domain_lower`: Lower bound of approximation domain
- `domain_upper`: Upper bound of approximation domain

Example:

```mlir
%result = math.sin %input {
  degree = 5 : i32,
  domain_lower = -3.14159 : f64,
  domain_upper = 3.14159 : f64
} : f32
```

## Binary Operation Handling

For binary operations, the pass applies when one operand is a constant:

```mlir
%c2 = arith.constant 2.0 : f32
%result = math.powf %x, %c2 : f32  // Can be approximated
```

## Benefits

- **FHE Compatibility**: Converts incompatible operations to FHE-friendly
  polynomials
- **Configurable Accuracy**: Allows control over approximation degree and domain
- **Comprehensive Coverage**: Supports a wide range of mathematical functions
- **Static Polynomials**: Generates compile-time polynomial coefficients

## Limitations

- **Approximation Error**: Results are approximate, not exact
- **Domain Restrictions**: Approximations are only valid within specified
  domains
- **Degree Trade-offs**: Higher degree improves accuracy but increases
  computation cost
- **Binary Operation Constraints**: Only applies to binary ops with one constant
  operand

## Approximation Quality

The quality of approximation depends on:

- **Polynomial Degree**: Higher degrees provide better accuracy
- **Domain Size**: Smaller domains allow better approximation
- **Function Smoothness**: Some functions approximate better than others

## Related Passes

- Often used early in FHE compilation pipelines
- May be followed by polynomial evaluation optimization passes
- Works well with constant folding to optimize polynomial coefficients
- Can be combined with other FHE-specific transformation passes
