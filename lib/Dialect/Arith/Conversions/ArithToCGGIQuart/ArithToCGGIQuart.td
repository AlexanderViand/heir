#ifndef LIB_DIALECT_ARITH_CONVERSIONS_ARITHTOCGGIQUART_TD_
#define LIB_DIALECT_ARITH_CONVERSIONS_ARITHTOCGGIQUART_TD_

include "mlir/Pass/PassBase.td"

/// **Developer Documentation: ArithToCGGIQuart Pass**
///
/// This pass implements precision decomposition for arithmetic operations,
/// converting high-precision (32-bit) arithmetic to lower-precision (8-bit)
/// CGGI operations for more efficient boolean circuit evaluation.
///
/// **Implementation Details:**
/// - Decomposes 32-bit operations into four 8-bit components
/// - Uses tensor operations to manage decomposed arithmetic values
/// - Stores 8-bit components in 16-bit integers to preserve carry information
/// - Implements carry propagation for multi-precision arithmetic
///
/// **Precision Strategy:**
/// - 32-bit → 4×8-bit decomposition for manageable circuit complexity
/// - LSB-first ordering in tensor representation
/// - 16-bit storage prevents carry bit loss
/// - Tensor dialect integration for multi-component operations
///
/// **Neural Network Applications:**
/// - Essential for TOSA neural network model compilation
/// - Enables efficient FHE inference through precision decomposition
/// - Works with multiply-accumulate optimization patterns
/// - Reduces circuit complexity compared to full-precision operations
///
/// **Based on MLIR Infrastructure:**
/// - Inspired by arith-emulate-wide-int pass from MLIR
/// - Uses established patterns for wide integer emulation
/// - Integrates with standard MLIR optimization infrastructure
///
/// **Extension Points:**
/// - Support for different decomposition strategies (16-bit, 4-bit)
/// - Alternative tensor representations for different backend targets
/// - Integration with specialized neural network optimization passes
def ArithToCGGIQuart : Pass<"arith-to-cggi-quart"> {
  let summary = "Lower `arith` to `cggi` dialect and divide each operation into smaller parts.";
    let description = [{
    This pass converts high precision arithmetic operations, i.e. operations on 32 bit integer,
    into a sequence of lower precision operations, i.e 8b operations.
    Currently, the pass splits the 32b integer into four 8b integers, using the tensor dialect.
    These smaller integers are stored in an 16b integer, so that we don't lose the carry information.
    This pass converts the `arith` dialect to the `cggi` dialect.

    Based on the `arith-emulate-wide-int` pass from the MLIR arith dialect.

    General assumption: the first element in the tensor is also the LSB element.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::tensor::TensorDialect",
    "mlir::heir::cggi::CGGIDialect",
  ];
}

#endif  // LIB_DIALECT_ARITH_CONVERSIONS_ARITHTOCGGIQUART_ARITHTOCGGIQUART_TD_
