#ifndef LIB_TRANSFORMS_LOWERORION_LOWERORION_TD_
#define LIB_TRANSFORMS_LOWERORION_LOWERORION_TD_

include "mlir/Pass/PassBase.td"

def LowerOrion : Pass<"lower-orion"> {
  let summary = "Lower Orion ops with explicit implementation styles";
  let description = [{
    Lowers Orion operations whose requested implementation style is an explicit
    compiler realization rather than an opaque backend operation.

    Currently supported:
    - `orion.linear_transform` with `orion.impl_style = "diagonal-basic"`
      lowers same-level transforms to explicit CKKS rotations, plaintext
      encodes, plaintext multiplications, and additions.

    Orion ops that request an opaque backend implementation remain unchanged.
  }];
  let dependentDialects = [
    "::mlir::heir::ckks::CKKSDialect",
    "::mlir::heir::lwe::LWEDialect",
    "::mlir::heir::orion::OrionDialect",
    "::mlir::tensor::TensorDialect",
  ];
}

#endif  // LIB_TRANSFORMS_LOWERORION_LOWERORION_TD_
