#ifndef LIB_DIALECT_SECRET_CONVERSIONS_SECRETTOBGV_SECRETTOBGV_TD_
#define LIB_DIALECT_SECRET_CONVERSIONS_SECRETTOBGV_SECRETTOBGV_TD_

include "mlir/Pass/PassBase.td"

/// **Developer Documentation: SecretToBGV Pass**
///
/// This pass implements the lowering from the high-level `secret` dialect to the
/// BGV (Brakerski-Gentry-Vaikuntanathan) FHE scheme dialect. It is a critical
/// component in the HEIR FHE compilation pipeline.
///
/// **Implementation Details:**
/// - Converts `secret.generic` operations to BGV ciphertext operations
/// - Handles arithmetic operations (add, mul, sub) on encrypted data
/// - Manages polynomial ring parameters and ciphertext encoding
/// - Integrates with RNS (Residue Number System) for multi-moduli operations
///
/// **Reusability Patterns:**
/// - Pattern-based conversion using MLIR's DialectConversion framework
/// - Type conversion utilities for secret-to-ciphertext type mappings
/// - Shared infrastructure with other Secret-to-FHE passes (CKKS, etc.)
///
/// **Extension Points:**
/// - Add new BGV operations by extending the conversion patterns
/// - Additional arithmetic operations can be supported by adding patterns
/// - Modify polynomial parameters through the poly-mod-degree option
/// - Integrate with new RNS configurations for different moduli chains
///
/// **Dependencies:**
/// - Requires Polynomial, ModArith, RNS, BGV, and LWE dialects
/// - Uses shared conversion utilities from Utils/ContextAwareConversion*
/// - Depends on proper secret operation distribution (secret-distribute-generic)
def SecretToBGV : Pass<"secret-to-bgv"> {
  let summary = "Lower `secret` to `bgv` dialect.";

  let description = [{
    This pass lowers an IR with `secret.generic` blocks containing arithmetic
    operations to operations on ciphertexts with the BGV dialect.

    The pass assumes that the `secret.generic` regions have been distributed
    through arithmetic operations so that only one ciphertext operation appears
    per generic block. It also requires that `canonicalize` was run so that
    non-secret values used are removed from the `secret.generic`'s block
    arguments.

    The pass requires that all types are tensors of a uniform shape matching the
    dimension of the ciphertext space specified by `poly-mod-degree`.

    **Technical Notes:**
    - Supports SIMD-style operations through tensor packing
    - Automatically manages noise growth through BGV operations
    - Preserves tensor semantics for vectorized FHE computations
    - Compatible with downstream LWE lowering for backend targeting
  }];

  let dependentDialects = [
    "mlir::heir::polynomial::PolynomialDialect",
    "mlir::heir::mod_arith::ModArithDialect",
    "mlir::heir::rns::RNSDialect",
    "mlir::heir::bgv::BGVDialect",
    "mlir::heir::lwe::LWEDialect",
  ];

  let options = [
    Option<"polyModDegree", "poly-mod-degree", "int",
           /*default=*/"1024", "Default degree of the cyclotomic polynomial "
           "modulus to use for ciphertext space.">,
  ];
}

#endif  // LIB_DIALECT_SECRET_CONVERSIONS_SECRETTOBGV_SECRETTOBGV_TD_
