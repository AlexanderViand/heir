#ifndef LIB_TRANSFORMS_LOWERUNPACK_LOWERUNPACK_TD_
#define LIB_TRANSFORMS_LOWERUNPACK_LOWERUNPACK_TD_

include "mlir/Pass/PassBase.td"

def LowerUnpack : Pass<"lower-unpack"> {
  let summary = "Lower tensor_ext.unpack to standard MLIR";
  let description = [{
  This pass lowers tensor_ext.unpack operations to equivalent standard MLIR operations.

  Developer Notes:
  - Converts HEIR-specific tensor_ext.unpack operations to standard MLIR tensor operations
  - Essential for lowering HEIR extensions to standard MLIR for broader compatibility
  - Enables use of standard MLIR optimization passes after custom tensor operations
  - Maintains semantic equivalence while using only standard tensor and arithmetic dialects
  - Bridge component: Connects HEIR-specific abstractions to standard MLIR ecosystem
  - Pipeline placement: After HEIR-specific optimizations but before standard MLIR passes
  - Reusable pattern: Template for lowering other custom tensor operations
  }];
  let dependentDialects = [
    "mlir::tensor::TensorDialect",
    "mlir::arith::ArithDialect",
  ];
}

#endif  // LIB_TRANSFORMS_LOWERUNPACK_LOWERUNPACK_TD_
