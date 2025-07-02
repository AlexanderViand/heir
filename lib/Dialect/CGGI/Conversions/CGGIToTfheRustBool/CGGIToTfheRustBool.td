#ifndef LIB_DIALECT_CGGI_CONVERSIONS_CGGITOTFHERUSTBOOL_CGGITOTFHERUSTBOOL_TD_
#define LIB_DIALECT_CGGI_CONVERSIONS_CGGITOTFHERUSTBOOL_CGGITOTFHERUSTBOOL_TD_

include "mlir/Pass/PassBase.td"

/// **Developer Documentation: CGGIToTfheRustBool Pass**
///
/// This pass implements specialized backend targeting from CGGI boolean operations
/// to TFHE-rs boolean-specific operations, providing an optimized path for
/// pure boolean computations with minimal overhead.
///
/// **Implementation Details:**
/// - Converts CGGI boolean operations to TFHE-rs boolean-specific API calls
/// - Specialized for pure boolean operations without arithmetic mixing
/// - Optimized for minimal memory usage and maximum throughput
/// - Handles boolean-only circuit patterns efficiently
///
/// **Boolean Specialization:**
/// - Focused on pure boolean operations (no mixed arithmetic)
/// - Optimized memory layout for boolean values
/// - Streamlined API calls for boolean-only operations
/// - Reduced overhead compared to general TFHE-rs targeting
///
/// **Performance Optimization:**
/// - Minimal memory footprint for boolean values
/// - Optimized for boolean circuit evaluation patterns
/// - Reduced API call overhead for boolean operations
/// - Specialized parameter handling for boolean-only workflows
def CGGIToTfheRustBool : Pass<"cggi-to-tfhe-rust-bool"> {
  let summary = "Lower `cggi` to `tfhe_rust_bool` dialect.";

  let description = [{
    This pass lowers the `cggi` dialect to `tfhe_rust_bool` dialect,
    providing a specialized backend targeting for pure boolean operations
    with optimized performance and minimal memory overhead.

    The pass is specifically designed for boolean-only computations,
    offering better performance than the general TFHE-rs targeting
    when working with pure boolean circuits.

    **Technical Notes:**
    - Optimized for boolean-only operations with minimal overhead
    - Specialized memory layout for boolean values
    - Streamlined API calls for pure boolean computation
    - Better performance than general TFHE-rs for boolean-only workloads
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::heir::cggi::CGGIDialect",
    "mlir::heir::lwe::LWEDialect",
    "mlir::heir::tfhe_rust_bool::TfheRustBoolDialect",
  ];
}

#endif  // LIB_DIALECT_CGGI_CONVERSIONS_CGGITOTFHERUSTBOOL_CGGITOTFHERUSTBOOL_TD_
