#ifndef LIB_DIALECT_LWE_CONVERSIONS_LWETOOPENFHE_LWETOOPENFHE_TD_
#define LIB_DIALECT_LWE_CONVERSIONS_LWETOOPENFHE_LWETOOPENFHE_TD_

include "mlir/Pass/PassBase.td"

/// **Developer Documentation: LWEToOpenfhe Pass**
///
/// This pass implements the backend targeting from generic LWE operations to
/// OpenFHE library-specific operations, enabling code generation for one of
/// the most widely used FHE libraries.
///
/// **Implementation Details:**
/// - Converts generic LWE operations to OpenFHE-specific API calls
/// - Handles both exact arithmetic (BGV) and approximate arithmetic (CKKS) modes
/// - Manages OpenFHE-specific parameter settings and optimizations
/// - Supports tensor operations for SIMD-packed ciphertext operations
///
/// **Backend Integration:**
/// - Generates code compatible with OpenFHE C++ API
/// - Handles OpenFHE context creation and parameter management
/// - Supports both serial and parallel OpenFHE configurations
/// - Integrates with OpenFHE's built-in optimization features
///
/// **Temporary Patterns:**
/// - Currently includes patterns for direct CKKS/BGV operations
/// - This is a transitional approach while LWE abstraction is completed
/// - Future versions will operate purely on LWE operations (TODO #1193)
///
/// **Reusability Patterns:**
/// - Template for other FHE library backends (Lattigo, SEAL, etc.)
/// - Uses standard MLIR code generation patterns
/// - Extensible framework for OpenFHE API evolution
///
/// **Extension Points:**
/// - Add support for new OpenFHE operations as they are introduced
/// - Implement OpenFHE-specific optimization patterns
/// - Add support for advanced OpenFHE features (GPU acceleration, etc.)
/// - Extend parameter tuning and security level management
///
/// **Code Generation Features:**
/// - Generates efficient OpenFHE C++ code
/// - Handles memory management for OpenFHE objects
/// - Supports both in-memory and serialized ciphertext operations
/// - Integrates with OpenFHE's parameter optimization
///
/// **Performance Considerations:**
/// - Leverages OpenFHE's built-in optimizations
/// - Supports OpenFHE's threading and parallelization
/// - Efficient memory usage patterns for large computations
/// - Compatible with OpenFHE's hardware acceleration features
///
/// **Dependencies:**
/// - Requires LWE, OpenFHE, and Tensor dialects
/// - May temporarily require CKKS and BGV dialects
/// - Generates code that links against OpenFHE library
def LWEToOpenfhe : Pass<"lwe-to-openfhe"> {
  let summary = "Lower `lwe` to `openfhe` dialect.";

  let description = [{
    This pass lowers the `lwe` dialect to `openfhe` dialect, enabling code
    generation for the OpenFHE library, one of the most widely used FHE
    implementations.

    The pass converts generic LWE operations to OpenFHE-specific API calls,
    handling both exact arithmetic (BGV) and approximate arithmetic (CKKS)
    modes. It generates efficient C++ code that integrates with OpenFHE's
    optimization features and parameter management.

    Currently, this also includes patterns that apply directly to `ckks` and
    `bgv` dialect operations as a transitional measure while the LWE
    abstraction layer is being completed.

    **Technical Notes:**
    - Generates OpenFHE C++ API calls with proper memory management
    - Supports both serial and parallel OpenFHE configurations
    - Handles OpenFHE context creation and parameter optimization
    - Compatible with tensor operations for SIMD-packed ciphertext operations

    TODO (#1193): investigate if the need for `ckks/bgv` patterns is permanent.
  }];

  let dependentDialects = [
    "mlir::heir::lwe::LWEDialect",
    "mlir::heir::openfhe::OpenfheDialect",
    "mlir::tensor::TensorDialect",
  ];
}

#endif  // LIB_DIALECT_LWE_CONVERSIONS_LWETOOPENFHE_LWETOOPENFHE_TD_
