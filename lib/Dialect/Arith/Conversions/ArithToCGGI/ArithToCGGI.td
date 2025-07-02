#ifndef LIB_DIALECT_ARITH_CONVERSIONS_ARITHTOCGGI_TD_
#define LIB_DIALECT_ARITH_CONVERSIONS_ARITHTOCGGI_TD_

include "mlir/Pass/PassBase.td"

/// **Developer Documentation: ArithToCGGI Pass**
///
/// This pass implements the conversion from standard arithmetic operations to
/// CGGI boolean operations, enabling arithmetic computation through boolean
/// circuit evaluation on encrypted data.
///
/// **Implementation Details:**
/// - Converts arithmetic operations to boolean circuit equivalents
/// - Maps integer arithmetic to gate-level boolean operations
/// - Handles bitwise decomposition of arithmetic operations
/// - Supports boolean circuit optimization for arithmetic algorithms
///
/// **Conversion Strategy:**
/// - Decomposes arithmetic operations into boolean gate sequences
/// - Uses boolean circuits to implement addition, multiplication, etc.
/// - Handles carry propagation and overflow detection
/// - Optimizes for efficient boolean gate evaluation
///
/// **Use Cases:**
/// - Boolean circuit implementations of arithmetic algorithms
/// - CGGI backend targeting for arithmetic computations
/// - Low-level arithmetic operations using boolean gates
/// - Integration with boolean circuit optimization tools
def ArithToCGGI : Pass<"arith-to-cggi"> {
  let summary = "Lower `arith` to `cggi` dialect.";

  let description = [{
    This pass lowers standard `arith` dialect operations to `cggi` dialect
    operations, converting arithmetic computations to boolean circuit
    implementations suitable for homomorphic evaluation.

    The pass enables arithmetic operations to be performed using boolean
    gate evaluation, which is efficient for certain FHE schemes like CGGI.

    **Technical Notes:**
    - Converts arithmetic to boolean gate sequences
    - Handles bitwise arithmetic implementation
    - Compatible with boolean circuit optimization passes
    - Suitable for CGGI-based FHE backends
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::heir::cggi::CGGIDialect",
  ];
}

#endif  // LIB_DIALECT_ARITH_CONVERSIONS_ARITHTOCGGI_ARITHTOCGGI_TD_
