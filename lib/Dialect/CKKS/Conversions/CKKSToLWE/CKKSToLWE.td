#ifndef LIB_DIALECT_CKKS_CONVERSIONS_CKKSTOLWE_CKKSTOLWE_TD_
#define LIB_DIALECT_CKKS_CONVERSIONS_CKKSTOLWE_CKKSTOLWE_TD_

include "mlir/Pass/PassBase.td"

/// **Developer Documentation: CKKSToLWE Pass**
///
/// This pass implements the abstraction layer between scheme-specific CKKS
/// operations and the generic LWE dialect, providing a common intermediate
/// representation for approximate arithmetic FHE operations.
///
/// **Implementation Details:**
/// - Converts CKKS-specific operations to generic LWE operations
/// - Preserves approximate arithmetic semantics while removing scheme details
/// - Enables backend-agnostic optimization and analysis for floating-point FHE
/// - Maintains compatibility with tensor-based SIMD operations
///
/// **Key Differences from BGVToLWE:**
/// - Handles approximate arithmetic instead of exact computation
/// - Preserves scale management semantics at the LWE level
/// - Manages rescaling operations and precision control
/// - Supports complex number and floating-point operation semantics
///
/// **Reusability Patterns:**
/// - Follows same pattern as BGVToLWE conversion
/// - Uses standard MLIR dialect conversion infrastructure
/// - Extensible pattern-based conversion system
///
/// **Extension Points:**
/// - Add support for new CKKS operations as they are introduced
/// - Implement "full" LWE lowering mode for complete abstraction
/// - Extend to handle CKKS-specific optimizations
/// - Add support for advanced scale management strategies
///
/// **Current Limitations:**
/// - Some CKKS-specific operations (e.g., rescale) remain unchanged
/// - Currently implements "common" lowering; "full" lowering planned
/// - Complex number semantics may be simplified during conversion
///
/// **Backend Compatibility:**
/// - Enables targeting multiple FHE libraries with CKKS support
/// - Provides uniform interface for approximate arithmetic backends
/// - Supports both packed and unpacked floating-point representations
///
/// **Dependencies:**
/// - Requires CKKS, LWE, and Tensor dialects
/// - Compatible with downstream backend conversion passes
/// - Integrates with LWE-level optimization and analysis passes
def CKKSToLWE : Pass<"ckks-to-lwe"> {
  let summary = "Lower `ckks` to `lwe` dialect.";

  let description = [{
    This pass lowers the `ckks` dialect to `lwe` dialect, providing a
    backend-agnostic intermediate representation for approximate arithmetic
    FHE operations.

    The pass abstracts away CKKS-specific details while preserving the
    essential semantics of approximate homomorphic operations. This enables
    the use of common optimization passes and supports multiple backend targets
    for floating-point FHE computations.

    Note that some scheme-specific operations (e.g., rescale) that
    have no direct analogue in the `lwe` dialect are left unchanged.

    **Technical Notes:**
    - Provides abstraction layer for backend-agnostic approximate arithmetic
    - Maintains tensor semantics for SIMD-style floating-point operations
    - Preserves scale management and precision control in LWE representation
    - Enables cross-backend optimization for floating-point FHE workloads

    TODO (#1193): support both "common" and "full" lwe lowering modes
  }];

  let dependentDialects = [
    "mlir::heir::ckks::CKKSDialect",
    "mlir::heir::lwe::LWEDialect",
    "mlir::tensor::TensorDialect",
  ];
}

#endif  // LIB_DIALECT_CKKS_CONVERSIONS_CKKSTOLWE_CKKSTOLWE_TD_
