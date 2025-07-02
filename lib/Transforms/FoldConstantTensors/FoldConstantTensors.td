#ifndef LIB_TRANSFORMS_FOLDCONSTANTTENSORS_FOLDCONSTANTTENSORS_TD_
#define LIB_TRANSFORMS_FOLDCONSTANTTENSORS_FOLDCONSTANTTENSORS_TD_

include "mlir/Pass/PassBase.td"

def FoldConstantTensors : Pass<"fold-constant-tensors"> {
  let summary = "This pass folds any constant tensors.";

  let dependentDialects = [
    "arith::ArithDialect"
  ];

  let description = [{
    This pass folds tensor operations on constants to new constants.

    The following folders are supported:
    * `tensor.insert` of a constant tensor
    * `tensor.collapse_shape` of a constant tensor

    Developer Notes:
    - Specialized constant folding for tensor operations not covered by standard canonicalization
    - Critical for FHE circuit optimization where constant tensors are common
    - Enables compile-time evaluation of tensor shape and content transformations
    - Reduces runtime overhead by precomputing constant tensor operations
    - Complements apply-folders pass with tensor-specific optimizations
    - Performance benefit: Eliminates runtime tensor operations when operands are constant
    - Pipeline placement: After loop unrolling and forwarding passes that expose constants
    - Reusable component: Folding patterns can be extended for other tensor operations
    - Memory optimization: Reduces intermediate tensor allocations for constant operations
  }];
}

#endif  // LIB_TRANSFORMS_FOLDCONSTANTTENSORS_FOLDCONSTANTTENSORS_TD_
