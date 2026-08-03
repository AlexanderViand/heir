#ifndef LIB_TRANSFORMS_LOWERAFFINEAPPLY_LOWERAFFINEAPPLY_TD_
#define LIB_TRANSFORMS_LOWERAFFINEAPPLY_LOWERAFFINEAPPLY_TD_

include "mlir/Pass/PassBase.td"

def LowerAffineApply : Pass<"lower-affine-apply"> {
  let summary = "Lower affine.apply ops to arithmetic, leaving affine.for intact";
  let description = [{
    Expands every `affine.apply` op into the equivalent sequence of `arith`
    ops (via MLIR's `expandAffineMap`), without touching `affine.for` or any
    other affine op.

    The scheme backend emitters (OpenFHE, Lattigo, ...) print `affine.for`
    directly but expect index arithmetic to already be expressed as scalar
    `arith` ops -- they have no printer for `affine.apply`. `affine.apply`
    ops appear when a non-unit-step / non-zero-lower-bound loop is normalized
    (the body's original index is recovered as `affine.apply (d0 * step + lb)`).
    `--lower-affine` would also rewrite `affine.for` into `scf.for`, which the
    emitters intentionally avoid, so this pass lowers only `affine.apply`.
  }];
  let dependentDialects = ["mlir::arith::ArithDialect"];
}

#endif  // LIB_TRANSFORMS_LOWERAFFINEAPPLY_LOWERAFFINEAPPLY_TD_
