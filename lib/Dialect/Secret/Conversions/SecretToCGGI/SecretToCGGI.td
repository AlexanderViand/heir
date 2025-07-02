#ifndef LIB_DIALECT_SECRET_CONVERSIONS_SECRETTOCGGI_SECRETTOCGGI_TD_
#define LIB_DIALECT_SECRET_CONVERSIONS_SECRETTOCGGI_SECRETTOCGGI_TD_

include "mlir/Pass/PassBase.td"

/// **Developer Documentation: SecretToCGGI Pass**
///
/// This pass implements the lowering from the high-level `secret` dialect to the
/// CGGI (Chillotti-Gama-Georgieva-Izabachène) framework for boolean circuit
/// evaluation over encrypted data.
///
/// **Implementation Details:**
/// - Converts `secret.generic` operations to CGGI boolean operations
/// - Handles truth tables and lookup tables (LUTs) for efficient boolean logic
/// - Transforms combinational logic operations from the `comb` dialect
/// - Supports bitwise operations and boolean circuit evaluation
///
/// **Key Features:**
/// - Optimized for boolean circuit evaluation with minimal noise growth
/// - Uses 3-input lookup tables (LUT3) for efficient arbitrary boolean functions
/// - Supports memref operations for bit manipulation
/// - Integrates with Yosys-optimized boolean circuits
///
/// **Reusability Patterns:**
/// - Pattern-based conversion for boolean operations
/// - Extensible framework for adding new boolean logic operations
/// - Shared infrastructure with other boolean-focused passes
///
/// **Extension Points:**
/// - Add support for new boolean operations and truth tables
/// - Extend LUT functionality for larger input sizes
/// - Implement custom boolean circuit optimization patterns
/// - Add support for additional combinational logic operations
///
/// **Dependencies:**
/// - Requires Comb, CGGI, MemRef, and Affine dialects
/// - Works closely with outputs from yosys-optimizer
/// - Depends on proper secret operation distribution
///
/// **Performance Considerations:**
/// - LUT3 operations provide efficient evaluation of arbitrary 3-input functions
/// - Minimizes bootstrapping operations through careful boolean logic selection
/// - Optimized for circuits with high boolean gate density
def SecretToCGGI : Pass<"secret-to-cggi"> {
  let summary = "Lower `secret` to `cggi` dialect.";

  let description = [{
    This pass lowers the `secret` dialect to `cggi` dialect for boolean
    circuit evaluation over encrypted data using the CGGI framework.

    The pass converts secret boolean operations to CGGI operations that can
    evaluate arbitrary boolean functions using lookup tables (LUTs) and
    basic boolean gates. It is particularly effective for circuits that
    have been optimized using the Yosys synthesis tool.

    **Technical Notes:**
    - Supports 3-input lookup tables for efficient boolean function evaluation
    - Handles combinational logic from the `comb` dialect
    - Optimized for boolean circuit evaluation with minimal noise growth
    - Integrates with memory operations for bit-level data manipulation
  }];

  let dependentDialects = [
    "mlir::heir::comb::CombDialect",
    "mlir::heir::cggi::CGGIDialect",
    "mlir::memref::MemRefDialect",
    "mlir::affine::AffineDialect",
  ];
}

#endif  // LIB_DIALECT_SECRET_CONVERSIONS_SECRETTOCGGI_SECRETTOCGGI_TD_
