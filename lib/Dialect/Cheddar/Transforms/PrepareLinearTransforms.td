#ifndef LIB_DIALECT_CHEDDAR_TRANSFORMS_PREPARELINEARTRANSFORMS_TD_
#define LIB_DIALECT_CHEDDAR_TRANSFORMS_PREPARELINEARTRANSFORMS_TD_

include "mlir/Pass/PassBase.td"

def CheddarPrepareLinearTransforms :
    Pass<"cheddar-prepare-linear-transforms", "mlir::ModuleOp"> {
  let summary = "Move linear-transform construction into split preprocessing";

  let description = [{
    Rewrites each `cheddar.linear_transform` in a split `__preprocessed`
    function into a `cheddar.prepare_linear_transform` in the paired
    `__preprocessing` function and a lightweight
    `cheddar.apply_prepared_linear_transform` in evaluation. The prepared
    object owns the encoded diagonal plaintexts and is returned alongside the
    ordinary split-preprocessing storage.
  }];

  let dependentDialects = [
    "mlir::heir::cheddar::CheddarDialect",
    "mlir::func::FuncDialect",
    "mlir::tensor::TensorDialect",
  ];
}

#endif  // LIB_DIALECT_CHEDDAR_TRANSFORMS_PREPARELINEARTRANSFORMS_TD_
