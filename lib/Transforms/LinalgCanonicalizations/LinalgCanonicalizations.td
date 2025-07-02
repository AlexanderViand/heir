#ifndef LIB_TRANSFORMS_LINALGCANONICALIZATIONS_LINALGCANONICALIZATIONS_TD_
#define LIB_TRANSFORMS_LINALGCANONICALIZATIONS_LINALGCANONICALIZATIONS_TD_

include "mlir/Pass/PassBase.td"

def LinalgCanonicalizations : Pass<"linalg-canonicalizations"> {
  let summary = "This pass canonicalizes the linalg.transpose operation of a constant into a transposed constant.";

  let dependentDialects = [
    "arith::ArithDialect"
  ];

  let description = [{
    This pass canonicalizes the linalg.transpose operation of a constant into a
    transposed constant.

    Developer Notes:
    - Compile-time optimization that evaluates constant transpose operations
    - Reduces runtime overhead by precomputing transpose of constant tensors
    - Essential for constant folding in tensor-heavy FHE computations
    - Enables better optimization opportunities by exposing constant structure
    - Reusable pattern: Logic can be extended to other constant linalg operations
    - Pipeline placement: After tensor constant creation but before arithmetic lowering
    - Performance benefit: Eliminates runtime transpose operations on known constants
  }];
}

#endif  // LIB_TRANSFORMS_LINALGCANONICALIZATIONS_LINALGCANONICALIZATIONS_TD_
