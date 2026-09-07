#ifndef LIB_DIALECT_CHEDDAR_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_CHEDDAR_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def CheddarBufferize : Pass<"cheddar-bufferize", "ModuleOp"> {
  let summary = "Run One-Shot Bufferize for Cheddar programs";
  let description = [{
    Converts Cheddar tensor function results to caller-provided destinations
    before bufferization, runs the standard empty-tensor elimination, then runs
    stock One-Shot Module Bufferization with function-boundary bufferization
    and identity layout maps. This exposes the final destination early enough
    for DPS producers to write into it directly. The bufferization memcpy hook
    rejects every remaining copy of a move-only Cheddar value, regardless of
    which participating dialect requested it.
  }];
}

#endif  // LIB_DIALECT_CHEDDAR_TRANSFORMS_PASSES_TD_
