#ifndef LIB_DIALECT_LWE_CONVERSIONS_LWETOCHEDDAR_LWETOCHEDDAR_TD_
#define LIB_DIALECT_LWE_CONVERSIONS_LWETOCHEDDAR_LWETOCHEDDAR_TD_

include "mlir/Pass/PassBase.td"

def LWEToCheddar : Pass<"lwe-to-cheddar"> {
  let summary = "Lower `lwe`/`ckks` to `cheddar` dialect.";

  let description = [{
    This pass lowers the `lwe` and `ckks` dialects to the `cheddar` dialect,
    targeting the CHEDDAR GPU FHE library.

    The pass threads a context, encoder, user interface, and multiplication
    key through function arguments following the same pattern as the
    LWE-to-Lattigo pass. Rotation/conjugation keys are not threaded as
    arguments: the `cheddar-to-emitc` lowering looks them up inline from the
    UserInterface at the use site.

    Linear transforms are intentionally not lowered by this pass. They must be
    decomposed before conversion until the separate Cheddar linear-transform
    support is available.
  }];

  let dependentDialects = [
    "mlir::heir::cheddar::CheddarDialect",
    "mlir::heir::ckks::CKKSDialect",
    "mlir::heir::lwe::LWEDialect",
    "mlir::tensor::TensorDialect",
    "mlir::bufferization::BufferizationDialect",
  ];

  let options = [
    Option<"enableMinKs", "enable-min-ks", "bool", "true",
           "Use scale-snu's minimum-key-switch linear-transform evaluation">
  ];
}

#endif  // LIB_DIALECT_LWE_CONVERSIONS_LWETOCHEDDAR_LWETOCHEDDAR_TD_
