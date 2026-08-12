#ifndef LIB_DIALECT_CHEDDAR_TRANSFORMS_FUSEOPS_TD_
#define LIB_DIALECT_CHEDDAR_TRANSFORMS_FUSEOPS_TD_

include "mlir/Pass/PassBase.td"

def CheddarFuseOps : Pass<"cheddar-fuse-ops"> {
  let summary = "Fuse Cheddar operations into compound GPU kernels";

  let description = [{
    Fuses Cheddar operation sequences that scale-snu/cheddar implements as a
    single GPU kernel:

    - mult + relinearize + rescale -> hmult (with rescale=true)
    - mult + relinearize -> hmult (with rescale=false)
    - mult + relinearize_rescale -> hmult (with rescale=true)
    - hrot + add -> hrot_add
    - hconj + add -> hconj_add

    The pass also moves relinearization before add_plain/sub_plain when needed;
    scale-snu/cheddar does not support those operations on an unrelinearized
    degree-3 ciphertext.
  }];

  let dependentDialects = [
    "mlir::heir::cheddar::CheddarDialect",
    "mlir::tensor::TensorDialect",
  ];
}

#endif  // LIB_DIALECT_CHEDDAR_TRANSFORMS_FUSEOPS_TD_
