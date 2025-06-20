#ifndef LIB_DIALECT_OPENFHE_CONVERSIONS_OPENFHETOSCHEME_OPENFHETOSCHEME_TD_
#define LIB_DIALECT_OPENFHE_CONVERSIONS_OPENFHETOSCHEME_OPENFHETOSCHEME_TD_

include "mlir/Pass/PassBase.td"

def OpenfheToScheme : Pass<"openfhe-to-scheme"> {
  let summary = "Convert the OpenFHE dialect to a scheme dialect";
  let description = [{
    Reverse the lowering performed by `--scheme-to-openfhe` by converting
    operations from the `openfhe` dialect back to the target scheme dialect
    (bgv, bfv or ckks). If the `scheme` option is not provided, the scheme
    is inferred from the module attributes.
  }];

  let options = [
    Option<"scheme", "scheme", "std::string", "", "Target scheme (bgv|bfv|ckks)">
  ];

  let dependentDialects = [
    "mlir::heir::openfhe::OpenfheDialect",
    "mlir::heir::bgv::BGVDialect",
    "mlir::heir::ckks::CKKSDialect",
    "mlir::heir::lwe::LWEDialect"
  ];
}

#endif // LIB_DIALECT_OPENFHE_CONVERSIONS_OPENFHETOSCHEME_OPENFHETOSCHEME_TD_
