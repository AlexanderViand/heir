#ifndef LIB_DIALECT_CKKS_CONVERSIONS_CKKSTOLWE_CKKSTOLWE_TD_
#define LIB_DIALECT_CKKS_CONVERSIONS_CKKSTOLWE_CKKSTOLWE_TD_

include "mlir/Pass/PassBase.td"

def CKKSToLWE : Pass<"ckks-to-lwe"> {
  let summary = "Lower `ckks` to `lwe` dialect.";

  let description = [{
    This pass lowers the `ckks` dialect to `lwe` dialect.
  }];

  let dependentDialects = [
    "mlir::heir::ckks::CKKSDialect",
    "mlir::heir::lwe::LWEDialect",
    "mlir::tensor::TensorDialect",
  ];
}

#endif  // LIB_DIALECT_CKKS_CONVERSIONS_CKKSTOLWE_CKKSTOLWE_TD_
