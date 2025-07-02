#ifndef LIB_DIALECT_LWE_CONVERSIONS_LWETOPOLYNOMIAL_LWETOPOLYNOMIAL_TD_
#define LIB_DIALECT_LWE_CONVERSIONS_LWETOPOLYNOMIAL_LWETOPOLYNOMIAL_TD_

include "mlir/Pass/PassBase.td"

/// **Developer Documentation: LWEToPolynomial Pass**
///
/// This pass implements the conversion from generic LWE operations to
/// polynomial-based representations, enabling mathematical analysis and
/// polynomial-level optimizations of FHE operations.
///
/// **Implementation Details:**
/// - Converts LWE operations to polynomial arithmetic operations
/// - Exposes the underlying polynomial structure of FHE operations
/// - Enables polynomial-level analysis and optimization passes
/// - Supports both univariate and multivariate polynomial representations
///
/// **Mathematical Foundation:**
/// - Maps ciphertext operations to polynomial ring operations
/// - Preserves algebraic structure for mathematical analysis
/// - Enables symbolic computation and algebraic optimization
/// - Supports polynomial evaluation and manipulation
///
/// **Use Cases:**
/// - Mathematical analysis of FHE algorithms
/// - Polynomial-level optimization and simplification
/// - Educational and research applications
/// - Algorithm prototyping with polynomial representations
///
/// **Reusability Patterns:**
/// - Provides foundation for polynomial-based analysis passes
/// - Extensible framework for polynomial optimization
/// - Template for mathematical backend targeting
///
/// **Extension Points:**
/// - Add support for advanced polynomial operations
/// - Implement polynomial optimization strategies
/// - Extend to support different polynomial bases
/// - Add integration with computer algebra systems
///
/// **Analysis Capabilities:**
/// - Enables degree analysis of polynomial operations
/// - Supports coefficient analysis and optimization
/// - Allows for symbolic computation on encrypted data operations
/// - Provides foundation for mathematical proofs and verification
///
/// **Dependencies:**
/// - Requires Polynomial, Tensor, and Random dialects
/// - May integrate with mathematical computation libraries
/// - Compatible with symbolic computation frameworks
def LWEToPolynomial : Pass<"lwe-to-polynomial"> {
  let summary = "Lower `lwe` to `polynomial` dialect.";

  let description = [{
    This pass lowers the `lwe` dialect to `polynomial` dialect, converting
    FHE operations to their underlying polynomial representations for
    mathematical analysis and polynomial-level optimizations.

    The pass exposes the algebraic structure of FHE operations by representing
    them as polynomial arithmetic operations. This enables mathematical
    analysis, symbolic computation, and polynomial-level optimization
    strategies that operate on the mathematical foundations of FHE.

    **Technical Notes:**
    - Maps ciphertext operations to polynomial ring operations
    - Preserves algebraic structure for mathematical analysis
    - Enables degree analysis and coefficient optimization
    - Supports both research and educational applications

    **Use Cases:**
    - Mathematical analysis and verification of FHE algorithms
    - Polynomial-level optimization and simplification
    - Research into FHE algorithm properties
    - Educational exploration of FHE mathematical foundations
  }];

  let dependentDialects = [
    "mlir::heir::polynomial::PolynomialDialect",
    "mlir::tensor::TensorDialect",
    "mlir::heir::random::RandomDialect"
  ];
}

#endif  // LIB_DIALECT_LWE_CONVERSIONS_LWETOPOLYNOMIAL_LWETOPOLYNOMIAL_TD_
