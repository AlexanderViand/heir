#ifndef LIB_DIALECT_CHEDDAR_TRANSFORMS_FREEINTERMEDIATES_TD_
#define LIB_DIALECT_CHEDDAR_TRANSFORMS_FREEINTERMEDIATES_TD_

include "mlir/Pass/PassBase.td"

def CheddarFreeIntermediates : Pass<"cheddar-free-intermediates"> {
  let summary = "Free intermediate CHEDDAR GPU buffers at their last use.";

  let description = [{
    Runs on the bufferized memref form (after one-shot-bufferize +
    buffer-results-to-out-params, before --convert-to-emitc). For each local
    `memref.alloc` whose element type is a CHEDDAR payload (Ciphertext/Plaintext,
    which own GPU memory), inserts a `memref.dealloc` right after the buffer's
    last use, so the device buffer is freed at last use instead of at the
    enclosing C++ function's scope-bound RAII cleanup. Without this, every
    intermediate stays live for the whole (often huge) function, so peak GPU
    memory is the sum of ALL intermediates and large models (criteoHELRM) OOM.

    This is a linear, use-list-based last-use insertion (CheddarToEmitC lowers
    each resulting payload dealloc to `v = {};`). It is used instead of MLIR's
    --ownership-based-buffer-deallocation, whose liveness/ownership analysis is
    superlinear and does not scale to the tens-of-thousands of locals in the
    generated preprocessing/eval functions. Only payload buffers are freed; plain
    float buffers are scope-bound stack arrays with nothing to free. Buffers that
    are out-param/function arguments are never `memref.alloc`s, so only true
    locals are freed.
  }];

  let dependentDialects = [
    "mlir::heir::cheddar::CheddarDialect",
    "mlir::memref::MemRefDialect",
  ];
}

#endif  // LIB_DIALECT_CHEDDAR_TRANSFORMS_FREEINTERMEDIATES_TD_
