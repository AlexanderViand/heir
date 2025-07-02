#ifndef LIB_TRANSFORMS_POLYNOMIALAPPROXIMATION_POLYNOMIALAPPROXIMATION_TD_
#define LIB_TRANSFORMS_POLYNOMIALAPPROXIMATION_POLYNOMIALAPPROXIMATION_TD_

include "mlir/Pass/PassBase.td"

// Pass for converting FHE-incompatible operations to polynomial approximations.
// This enables FHE computation of transcendental and complex mathematical functions
// by replacing them with polynomials that use only addition and multiplication.
//
// IMPLEMENTATION DETAILS:
// - Comprehensive coverage of math and arith dialect operations
// - Polynomial coefficient generation for various mathematical functions
// - Integration with polynomial dialect for evaluation operations
// - Configurable approximation parameters (degree, domain) via operation attributes
//
// REUSABILITY PATTERNS:
// - Function approximation framework can be extended to new operations
// - Polynomial generation can be adapted for different accuracy requirements
// - Domain specification system can be customized for application needs
// - Can be generalized to other computational models with limited operation sets
//
// DESIGN CONSIDERATIONS:
// - Balances approximation accuracy against computational complexity
// - Designed for FHE schemes that only support arithmetic operations
// - Handles both unary and binary operations with appropriate constraints
// - Enables complex mathematical computations in privacy-preserving contexts
def PolynomialApproximation : Pass<"polynomial-approximation"> {
  let summary = "Approximate ops by polynomials";
  let description = [{
    This pass replaces certain operations that are incompatible
    with the FHE computational model with polynomial approximations.

    The pass applies to the following ops in the `math` dialect. When the
    op is binary, the pass applies when one op is the result of an
    `arith.constant` which is scalar-valued or a splatted tensor.

    - `absf`
    - `acos`
    - `acosh`
    - `asin`
    - `asinh`
    - `atan2`
    - `atan`
    - `atanh`
    - `cbrt`
    - `ceil`
    - `copysign`
    - `cos`
    - `cosh`
    - `erf`
    - `erfc`
    - `exp2`
    - `exp`
    - `expm1`
    - `floor`
    - `fpowi`
    - `log10`
    - `log1p`
    - `log2`
    - `log`
    - `powf`
    - `round`
    - `roundeven`
    - `rsqrt`
    - `sin`
    - `sinh`
    - `sqrt`
    - `tan`
    - `tanh`
    - `trunc`

    The following ops in the `arith` dialect are also supported:

    - `maxf`
    - `maxnumf`
    - `minf`
    - `minnumf`

    These ops are replaced with `polynomial.eval` ops with a static polynomial
    attribute.

    (* example filepath=tests/Transforms/polynomial_approximation/doctest.mlir *)
  }];
  let dependentDialects = [
    "mlir::heir::polynomial::PolynomialDialect"
  ];
}

#endif  // LIB_TRANSFORMS_POLYNOMIALAPPROXIMATION_POLYNOMIALAPPROXIMATION_TD_
