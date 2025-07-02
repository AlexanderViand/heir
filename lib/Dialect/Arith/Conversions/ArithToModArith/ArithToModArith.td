#ifndef LIB_DIALECT_MODARITH_CONVERSIONS_ARITHTOMODARITH_ARITHTOMODARITH_TD_
#define LIB_DIALECT_MODARITH_CONVERSIONS_ARITHTOMODARITH_ARITHTOMODARITH_TD_

include "lib/Utils/DRR/Utils.td"
include "lib/Dialect/ModArith/IR/ModArithOps.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/Pass/PassBase.td"

/// **Developer Documentation: ArithToModArith Pass**
///
/// This pass implements the conversion from standard arithmetic operations to
/// modular arithmetic, enabling neural network compilation and preparing for
/// multiply-accumulate optimization in FHE pipelines.
///
/// **Implementation Details:**
/// - Converts standard arithmetic operations to modular equivalents
/// - Configurable modulus for different arithmetic requirements
/// - Essential preprocessing for neural network model compilation
/// - Enables efficient pattern detection for MAC operations
///
/// **Neural Network Focus:**
/// - Critical component in TOSA neural network compilation pipeline
/// - Prepares operations for multiply-accumulate pattern detection
/// - Enables precision decomposition for efficient FHE evaluation
/// - Supports large-to-small precision operation lowering
///
/// **Modulus Management:**
/// - Automatic modulus selection based on integer types
/// - Custom modulus support for specific requirements
/// - Well-defined overflow behavior through modular arithmetic
/// - Precision control for downstream optimization
///
/// **Pipeline Integration:**
/// - First step in neural network to FHE compilation
/// - Enables find-mac pass for MAC pattern detection
/// - Prepares for precision decomposition passes
/// - Compatible with TOSA dialect input operations
///
/// **Extension Points:**
/// - Support for different modulus selection strategies
/// - Integration with additional neural network optimization patterns
/// - Custom precision handling for specialized applications
def ArithToModArith : Pass<"arith-to-mod-arith", "ModuleOp"> {
  let summary = "Lower standard `arith` to `mod-arith`.";

  let description = [{
    This pass lowers the `arith` dialect to their `mod-arith` equivalents.

    The arith-to-mod-arith pass is required to lower a neural network TOSA
    model to a CGGI backend. This pass will transform the operations to the
    mod-arith dialect, where the find-mac pass can be used to convert
    consecutive multiply addition operations into a single operation. In a
    later pass, these large precision MAC operations (typically
    64 or 32-bit) will be lowered into small precision (8 or 4b) operations
    that can be mapped to CGGI operations.   }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::heir::mod_arith::ModArithDialect",
  ];

  let options = [
    Option<"modulus", "modulus", "int64_t",
           /*default=*/"0", "Modulus to use for the mod-arith dialect."
           " If not specified, the pass will use the natural modulus for that integer type">,
  ];
}

#endif  // LIB_DIALECT_MODARITH_CONVERSIONS_ARITHTOMODARITH_ARITHTOMODARITH_TD_
