#ifndef LIB_TRANSFORMS_RAISECKKS_RAISECKKS_TD_
#define LIB_TRANSFORMS_RAISECKKS_RAISECKKS_TD_

include "mlir/Pass/PassBase.td"

def RaiseCKKS : Pass<"raise-ckks"> {
  let summary = "Raise CKKS/Orion-level IR back to arith-level secret IR";
  let description = [{
    This pass converts CKKS-level IR (with LWE ciphertext types, CKKS ops,
    and Orion structured ops) back to arith-level IR inside `secret.generic`
    regions.

    The raised IR can then be re-managed through the standard CKKS pipeline
    (`--mlir-to-ckks` or `--torch-linalg-to-ckks`) to target any backend.

    Op conversions:
    - `orion.chebyshev` -> `polynomial.eval` (Chebyshev polynomial)
    - `orion.linear_transform` -> `linalg.matvec` (reconstructed matrix)
    - `ckks.add` -> `arith.addf`
    - `ckks.mul` -> `arith.mulf`
    - `ckks.rotate` -> `tensor_ext.rotate`
    - `ckks.add_plain` / `ckks.sub_plain` -> `arith.addf` / `arith.subf`
    - `ckks.mul_plain` -> `arith.mulf`
    - `ckks.rescale`, `ckks.relinearize`, `ckks.negate` -> stripped or converted

    Type conversions:
    - `!lwe.lwe_ciphertext<...>` -> `!secret.secret<tensor<Nxf64>>`
    - `!lwe.lwe_plaintext<...>` -> eliminated (folded into arith ops)

    Module attribute conversions:
    - `ckks.schemeParam` -> stripped (parameter generation will re-derive)
    - `scheme.ckks` -> preserved
  }];
  let dependentDialects = [
    "mlir::heir::secret::SecretDialect",
    "mlir::heir::polynomial::PolynomialDialect",
    "mlir::heir::tensor_ext::TensorExtDialect",
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect",
  ];
}

#endif  // LIB_TRANSFORMS_RAISECKKS_RAISECKKS_TD_
