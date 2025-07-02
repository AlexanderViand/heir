#ifndef LIB_DIALECT_POLYNOMIAL_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_POLYNOMIAL_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def PolyMulToNTT : Pass<"convert-polynomial-mul-to-ntt"> {
  let summary = "Rewrites polynomial operations to their NTT equivalents";
  let description = [{
    // DEVELOPER NOTE: This pass implements polynomial multiplication optimization using the
    // Number Theoretic Transform (NTT), which provides O(n log n) complexity instead of O(n²)
    // for direct convolution. This is a critical optimization for FHE schemes that rely heavily
    // on polynomial operations.
    //
    // Implementation details:
    // - Pattern-based rewriting using MLIR's rewrite framework
    // - Transforms polynomial.mul operations to NTT-based sequences
    // - Validates mathematical prerequisites (suitable modulus, ring size, roots of unity)
    // - Preserves semantic equivalence while improving algorithmic complexity
    //
    // NTT transformation sequence:
    // 1. Forward NTT on both operands: a(x) → A, b(x) → B (O(n log n) each)
    // 2. Pointwise multiplication: C = A ⊙ B (O(n))
    // 3. Inverse NTT: C → c(x) (O(n log n))
    // Total complexity: O(n log n) vs O(n²) for direct multiplication
    //
    // Mathematical requirements:
    // - Prime modulus q with q ≡ 1 (mod 2n) to ensure n-th roots of unity exist
    // - Ring size n typically power of 2 for optimal NTT efficiency
    // - Primitive n-th root of unity ω in the coefficient field
    //
    // Reusability considerations:
    // - Pattern-based design allows easy extension to other polynomial operations
    // - Mathematical validation framework can be reused for other number-theoretic optimizations
    // - NTT infrastructure benefits other cryptographic polynomial operations
    // - Backend-agnostic transformation enables optimization across different targets
    //
    // Performance implications:
    // - Asymptotic improvement crucial for large polynomial degrees (common in FHE)
    // - Memory access patterns in NTT are more cache-friendly than direct convolution
    // - Enables hardware acceleration on platforms with NTT-optimized instructions
    // - Parallelization opportunities in both forward/inverse NTT and pointwise operations
    Applies a rewrite pattern to convert polynomial multiplication to the
    equivalent using the number-theoretic transforms (NTT) when possible.

    Polynomial multiplication can be rewritten as polynomial.NTT
    on each operand, followed by modulo elementwise multiplication of the
    point-value representation and then the inverse-NTT back to coefficient
    representation.
  }];
  let dependentDialects = ["mlir::heir::polynomial::PolynomialDialect", "heir::mod_arith::ModArithDialect"];
}

#endif  // LIB_DIALECT_POLYNOMIAL_TRANSFORMS_PASSES_TD_
