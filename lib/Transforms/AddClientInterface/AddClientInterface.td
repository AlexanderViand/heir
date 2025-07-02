#ifndef LIB_TRANSFORMS_ADDCLIENTINTERFACE_ADDCLIENTINTERFACE_TD_
#define LIB_TRANSFORMS_ADDCLIENTINTERFACE_ADDCLIENTINTERFACE_TD_

include "mlir/Pass/PassBase.td"

def AddClientInterface : Pass<"add-client-interface"> {
  let summary = "Add client interfaces to secret functions";
  let description = [{
  This pass adds encrypt and decrypt functions for each compiled function in the
  IR. These functions maintain the same interface as the original function,
  while the compiled function may lose some of this information by the lowerings
  to ciphertext types (e.g., a scalar ciphertext, when lowered through RLWE schemes,
  must be encoded as a tensor).

  This pass occurs at the secret level, which is necessary because some
  backends like the plaintext backend don't actually encrypt, but still require
  the ciphertext layout/packing logic to convert cleartexts to plaintexts.

  Developer Notes:
  - Critical for FHE client-server separation and secure computation workflows
  - Generates encode/decode functions that handle data layout transformations
  - Preserves original function interface while accommodating ciphertext packing requirements
  - Backend-agnostic design: Works with both encryption and plaintext backends
  - Interface preservation: Maintains type signatures for client-side usage
  - Layout management: Handles conversion between scalar and packed tensor representations
  - Security consideration: Ensures proper data encoding for encrypted computation
  - Pipeline placement: Late in compilation after most optimizations but before backend lowering
  }];
  let dependentDialects = ["mlir::heir::secret::SecretDialect", "mlir::heir::tensor_ext::TensorExtDialect"];

  let options = [
    Option<
      "ciphertextSize",
      "ciphertext-size",
      "int",
      /*default=*/"1024",
      "Power of two length of the ciphertexts the data is packed in."
    >,
    Option<
      "enableLayoutAssignment",
      "enable-layout-assignment",
      "bool",
      /*default=*/"true",
      "If false, skips the emission of layout assignment operations, essentially assuming that the input was already using correctly (ciphertext-)sized tensors."
    >
  ];
}

#endif  // LIB_TRANSFORMS_ADDCLIENTINTERFACE_ADDCLIENTINTERFACE_TD_
