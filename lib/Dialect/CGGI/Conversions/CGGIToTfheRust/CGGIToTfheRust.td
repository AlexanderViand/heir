#ifndef LIB_DIALECT_CGGI_CONVERSIONS_CGGITOTFHERUST_CGGITOTFHERUST_TD_
#define LIB_DIALECT_CGGI_CONVERSIONS_CGGITOTFHERUST_CGGITOTFHERUST_TD_

include "mlir/Pass/PassBase.td"

/// **Developer Documentation: CGGIToTfheRust Pass**
///
/// This pass implements the backend targeting from CGGI boolean operations to
/// TFHE-rs library-specific operations, enabling code generation for the
/// high-performance Rust-based FHE library optimized for boolean circuits.
///
/// **Implementation Details:**
/// - Converts CGGI boolean operations to TFHE-rs API calls
/// - Handles lookup tables (LUTs) and boolean gate operations
/// - Manages TFHE-rs parameter settings and optimization
/// - Supports efficient boolean circuit evaluation patterns
///
/// **TFHE-rs Integration:**
/// - Generates code compatible with TFHE-rs Rust API
/// - Handles TFHE-rs context and key management
/// - Supports TFHE-rs's optimized boolean operations
/// - Integrates with Rust's memory safety and performance features
///
/// **Boolean Circuit Specialization:**
/// - Optimized for boolean gate evaluation over encrypted data
/// - Supports efficient LUT3 operations for arbitrary 3-input functions
/// - Handles bitwise operations and combinational logic
/// - Enables high-performance encrypted boolean computation
///
/// **Rust Language Features:**
/// - Leverages Rust's memory safety and zero-cost abstractions
/// - Supports Rust's ownership model for safe resource management
/// - Integrates with Rust's package ecosystem (Cargo)
/// - Provides safe, efficient generated code
///
/// **Reusability Patterns:**
/// - Template for other boolean-focused FHE backends
/// - Uses standard MLIR code generation patterns
/// - Extensible framework for TFHE-rs API evolution
///
/// **Extension Points:**
/// - Add support for new TFHE-rs operations as they are introduced
/// - Implement TFHE-rs-specific optimization patterns
/// - Add support for advanced TFHE-rs features (multi-threading, etc.)
/// - Extend integration with Rust ecosystem tools
///
/// **Performance Considerations:**
/// - Leverages TFHE-rs's highly optimized boolean operations
/// - Supports Rust's efficient memory management
/// - Compatible with parallel computation patterns
/// - Optimized for high-throughput boolean circuit evaluation
///
/// **Dependencies:**
/// - Requires Arith, CGGI, LWE, and TfheRust dialects
/// - Generates code that depends on TFHE-rs Rust crate
/// - Compatible with Rust build toolchain and Cargo
def CGGIToTfheRust : Pass<"cggi-to-tfhe-rust"> {
  let summary = "Lower `cggi` to `tfhe_rust` dialect.";

  let description = [{
    This pass lowers the `cggi` dialect to `tfhe_rust` dialect, enabling
    code generation for the TFHE-rs library, a high-performance Rust-based
    FHE implementation optimized for boolean circuit evaluation.

    The pass converts CGGI boolean operations to TFHE-rs-specific Rust API
    calls, supporting efficient evaluation of boolean circuits over encrypted
    data. It generates safe, performant Rust code that leverages TFHE-rs's
    optimizations for boolean operations.

    **Technical Notes:**
    - Generates TFHE-rs Rust API calls with memory safety guarantees
    - Supports efficient LUT operations for boolean function evaluation
    - Handles TFHE-rs context creation and parameter management
    - Compatible with boolean circuit optimization patterns
    - Integrates with Rust's package ecosystem and build toolchain

    **Boolean Circuit Focus:**
    - Optimized for high-performance boolean gate evaluation
    - Supports combinational logic and bitwise operations
    - Enables efficient encrypted boolean computation
    - Compatible with Yosys-optimized boolean circuits
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::heir::cggi::CGGIDialect",
    "mlir::heir::lwe::LWEDialect",
    "mlir::heir::tfhe_rust::TfheRustDialect",
  ];
}

#endif  // LIB_DIALECT_CGGI_CONVERSIONS_CGGITOTFHERUST_CGGITOTFHERUST_TD_
