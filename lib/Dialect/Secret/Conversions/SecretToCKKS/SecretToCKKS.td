#ifndef LIB_DIALECT_SECRET_CONVERSIONS_SECRETTOCKKS_SECRETTOCKKS_TD_
#define LIB_DIALECT_SECRET_CONVERSIONS_SECRETTOCKKS_SECRETTOCKKS_TD_

include "mlir/Pass/PassBase.td"

/// **Developer Documentation: SecretToCKKS Pass**
///
/// This pass implements the lowering from the high-level `secret` dialect to the
/// CKKS (Cheon-Kim-Kim-Song) FHE scheme dialect, enabling approximate arithmetic
/// on encrypted floating-point data.
///
/// **Implementation Details:**
/// - Converts `secret.generic` operations to CKKS ciphertext operations
/// - Handles floating-point arithmetic with controlled precision loss
/// - Manages scale factors for fixed-point encoding of real numbers
/// - Supports SIMD operations through complex number packing
///
/// **Key Differences from BGV:**
/// - Approximate arithmetic instead of exact computation
/// - Built-in support for complex numbers and real arithmetic
/// - Scale management for controlling precision vs. noise trade-offs
/// - Rescaling operations to manage ciphertext modulus growth
///
/// **Reusability Patterns:**
/// - Shares conversion infrastructure with SecretToBGV pass
/// - Uses common type conversion utilities for secret-to-ciphertext mappings
/// - Pattern-based approach allows easy extension for new CKKS operations
///
/// **Extension Points:**
/// - Add support for additional floating-point operations
/// - Implement custom scale management strategies
/// - Extend polynomial parameter configurations
/// - Add support for complex number operations
///
/// **Dependencies:**
/// - Requires Polynomial, ModArith, RNS, CKKS, LWE, and Tensor dialects
/// - Uses shared conversion utilities from Utils/ContextAwareConversion*
/// - Depends on proper secret operation distribution
def SecretToCKKS : Pass<"secret-to-ckks"> {
  let summary = "Lower `secret` to `ckks` dialect.";

  let description = [{
    This pass lowers an IR with `secret.generic` blocks containing arithmetic
    operations to operations on ciphertexts with the CKKS dialect.

    The pass assumes that the `secret.generic` regions have been distributed
    through arithmetic operations so that only one ciphertext operation appears
    per generic block. It also requires that `canonicalize` was run so that
    non-secret values used are removed from the `secret.generic`'s block
    arguments.

    The pass requires that all types are tensors of a uniform shape matching the
    dimension of the ciphertext space specified by `poly-mod-degree`.

    **Technical Notes:**
    - Enables approximate homomorphic computation on real numbers
    - Automatically manages scale factors for precision control
    - Supports SIMD-packed operations for vectorized computation
    - Compatible with machine learning inference workloads
  }];

  let dependentDialects = [
    "mlir::heir::polynomial::PolynomialDialect",
    "mlir::heir::mod_arith::ModArithDialect",
    "mlir::heir::rns::RNSDialect",
    "mlir::heir::ckks::CKKSDialect",
    "mlir::heir::lwe::LWEDialect",
    "mlir::tensor::TensorDialect",
  ];

  let options = [
    Option<"polyModDegree", "poly-mod-degree", "int",
           /*default=*/"1024", "Default degree of the cyclotomic polynomial "
           "modulus to use for ciphertext space.">,
  ];
}

#endif  // LIB_DIALECT_SECRET_CONVERSIONS_SECRETTOCKKS_SECRETTOCKKS_TD_
