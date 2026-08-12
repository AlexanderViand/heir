#ifndef LIB_DIALECT_PREPROCESSING_CONVERSIONS_PREPROCESSINGTOCHEDDAR_PREPROCESSINGTOCHEDDAR_TD_
#define LIB_DIALECT_PREPROCESSING_CONVERSIONS_PREPROCESSINGTOCHEDDAR_PREPROCESSINGTOCHEDDAR_TD_

include "mlir/Pass/PassBase.td"

def PreprocessingToCheddar : Pass<"preprocessing-to-cheddar", "mlir::ModuleOp"> {
  let summary = "Lower Preprocessing ops to memrefs of Cheddar plaintexts";
  let description = [{
    Converts `preprocessing.storage` to a `memref` of the scalar
    `!cheddar.plaintext` element and lowers `empty`, `store`, and `load`.

    The Cheddar dialect represents a single plaintext as a rank-0
    `tensor<!cheddar.plaintext>`. A memref element type cannot be a tensor, so
    the storage memref holds the unwrapped scalar `!cheddar.plaintext`, and
    store/load bridge via `tensor.extract` / `tensor.from_elements`.

    This pass must run after `--lwe-to-cheddar`, which converts storage element
    types from `lwe.lwe_plaintext` to `tensor<!cheddar.plaintext>`.
  }];
  let dependentDialects = [
    "mlir::bufferization::BufferizationDialect",
    "mlir::memref::MemRefDialect",
    "mlir::tensor::TensorDialect",
    "mlir::arith::ArithDialect",
    "mlir::affine::AffineDialect",
    "mlir::heir::cheddar::CheddarDialect"
  ];
}

#endif  // LIB_DIALECT_PREPROCESSING_CONVERSIONS_PREPROCESSINGTOCHEDDAR_PREPROCESSINGTOCHEDDAR_TD_
