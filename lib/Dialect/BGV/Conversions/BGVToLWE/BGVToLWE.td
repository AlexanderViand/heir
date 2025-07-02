#ifndef LIB_DIALECT_BGV_CONVERSIONS_BGVTOLWE_BGVTOLWE_TD_
#define LIB_DIALECT_BGV_CONVERSIONS_BGVTOLWE_BGVTOLWE_TD_

include "mlir/Pass/PassBase.td"

/// **Developer Documentation: BGVToLWE Pass**
///
/// This pass implements the abstraction layer between scheme-specific BGV
/// operations and the generic LWE dialect, providing a common intermediate
/// representation for multiple FHE backends.
///
/// **Implementation Details:**
/// - Converts BGV-specific operations to generic LWE operations
/// - Preserves semantics while removing scheme-specific details
/// - Enables backend-agnostic optimization and analysis passes
/// - Maintains compatibility with tensor-based SIMD operations
///
/// **Abstraction Strategy:**
/// - Maps BGV arithmetic operations to corresponding LWE operations
/// - Preserves noise management semantics at the LWE level
/// - Removes BGV-specific parameter details while maintaining functionality
/// - Enables common optimization patterns across different FHE schemes
///
/// **Reusability Patterns:**
/// - Provides template for other scheme-to-LWE conversions (CKKS, etc.)
/// - Uses standard MLIR dialect conversion infrastructure
/// - Extensible pattern-based conversion system
///
/// **Extension Points:**
/// - Add support for new BGV operations as they are introduced
/// - Implement "full" LWE lowering mode for complete abstraction
/// - Extend to handle additional scheme-specific optimizations
/// - Add support for cross-scheme operation compatibility
///
/// **Current Limitations:**
/// - Some BGV-specific operations (e.g., modswitch) remain unchanged
/// - Currently implements "common" lowering; "full" lowering planned
/// - Scheme-specific optimizations may be lost during conversion
///
/// **Backend Compatibility:**
/// - Enables targeting multiple FHE libraries (OpenFHE, Lattigo, etc.)
/// - Provides uniform interface for backend-specific passes
/// - Supports both packed and unpacked ciphertext representations
///
/// **Dependencies:**
/// - Requires BGV, LWE, and Tensor dialects
/// - Compatible with downstream backend conversion passes
/// - Integrates with LWE-level optimization and analysis passes
def BGVToLWE : Pass<"bgv-to-lwe"> {
  let summary = "Lower `bgv` to `lwe` dialect.";

  let description = [{
    This pass lowers the `bgv` dialect to `lwe` dialect, providing a
    backend-agnostic intermediate representation for FHE operations.

    The pass abstracts away BGV-specific details while preserving the
    essential semantics of homomorphic operations. This enables the use
    of common optimization passes and supports multiple backend targets.

    Note that some scheme-specific operations (e.g., modswitch) that
    have no direct analogue in the `lwe` dialect are left unchanged.

    **Technical Notes:**
    - Provides abstraction layer for backend-agnostic FHE compilation
    - Maintains tensor semantics for SIMD-style operations
    - Preserves noise growth characteristics in LWE representation
    - Enables cross-backend optimization and analysis

    TODO (#1193): support both "common" and "full" lwe lowering modes
  }];

  let dependentDialects = [
    "mlir::heir::bgv::BGVDialect",
    "mlir::heir::lwe::LWEDialect",
    "mlir::tensor::TensorDialect",
  ];
}

#endif  // LIB_DIALECT_BGV_CONVERSIONS_BGVTOLWE_BGVTOLWE_TD_
