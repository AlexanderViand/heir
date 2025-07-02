#ifndef LIB_TRANSFORMS_DROPUNITDIMS_DROPUNITDIMS_TD_
#define LIB_TRANSFORMS_DROPUNITDIMS_DROPUNITDIMS_TD_

include "mlir/Pass/PassBase.td"

def DropUnitDims : Pass<"drop-unit-dims"> {
  let summary = "Drops unit dimensions from linalg ops.";
  let description = [{
  This pass converts `linalg` whose operands have unit dimensions
  in their types to specialized ops that drop these unit dimensions.

  For example, a `linalg.matmul` whose RHS has type `tensor<32x1xi32>` is
  converted to a `linalg.matvec` op on the underlying `tensor<32xi32>`.

  Developer Notes:
  - Specializes linalg operations to more efficient forms when unit dimensions are present
  - Critical optimization for FHE where matrix-vector operations are common
  - Enables use of specialized kernels for lower-dimensional operations
  - Reduces computational complexity by eliminating trivial dimensions
  - Pattern example: matmul(NxM, Mx1) → matvec(NxM, M) with better performance characteristics
  - Reusable transformation: Logic applicable to other operations with unit dimension specializations
  - Pipeline placement: After tensor shape analysis but before lowering to computation kernels
  }];
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
  ];
}

#endif  // LIB_TRANSFORMS_DROPUNITDIMS_DROPUNITDIMS_TD_
