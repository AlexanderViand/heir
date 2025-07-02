#ifndef LIB_TRANSFORMS_TENSORLINALGTOAFFINELOOPS_TENSORLINALGTOAFFINELOOPS_TD_
#define LIB_TRANSFORMS_TENSORLINALGTOAFFINELOOPS_TENSORLINALGTOAFFINELOOPS_TD_

include "mlir/Pass/PassBase.td"

// Pass extending standard linalg-to-affine-loops for tensor-semantic operations.
// This specialized conversion focuses on layout assignment operations and
// maintains tensor semantics while converting to explicit affine loop structures.
//
// IMPLEMENTATION DETAILS:
// - Extension of MLIR's standard convert-linalg-to-affine-loops pass
// - Special handling for tensor_ext.assign_layout operations
// - Tensor semantic preservation during loop conversion
// - Integration with HEIR's layout propagation system
//
// REUSABILITY PATTERNS:
// - Conversion framework can be extended to other tensor dialect operations
// - Layout assignment patterns can be adapted for different layout schemes
// - Affine loop generation can be customized for different optimization goals
// - Can be generalized to other tensor-to-loop conversion scenarios
//
// DESIGN CONSIDERATIONS:
// - Maintains compatibility with standard linalg conversion patterns
// - Designed specifically for HEIR's tensor layout system
// - Focuses on enabling subsequent affine optimizations
// - Balances explicit loop representation with semantic preservation
def TensorLinalgToAffineLoops : Pass<"tensor-linalg-to-affine-loops"> {
  let summary = "A port of convert-linalg-to-affine-loops for loops with tensor semantics";
  let description = [{
  This pass primarily exists to support the conversion of `linalg.generic`
  operations that implement `tensor_ext.assign_layout` ops.
  }];
  let dependentDialects = [
    "::mlir::affine::AffineDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::tensor::TensorDialect",
  ];
}

#endif  // LIB_TRANSFORMS_TENSORLINALGTOAFFINELOOPS_TENSORLINALGTOAFFINELOOPS_TD_
