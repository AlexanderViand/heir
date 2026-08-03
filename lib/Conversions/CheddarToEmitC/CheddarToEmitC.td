#ifndef LIB_CONVERSIONS_CHEDDARTOEMITC_CHEDDARTOEMITC_TD_
#define LIB_CONVERSIONS_CHEDDARTOEMITC_CHEDDARTOEMITC_TD_

include "mlir/Pass/PassBase.td"

def CheddarToEmitC : Pass<"cheddar-emitc-boundary"> {
  let summary = "Fix up EmitC function boundaries for the CHEDDAR move-only ABI.";

  let description = [{
    Runs after `--convert-to-emitc` (which lowers the cheddar/arith/scf/memref
    ops via their `ConvertToEmitCPatternInterface` implementations). The cheddar
    payload types are move-only and live in `emitc.lvalue`/`emitc.array` local
    buffers, but a function cannot carry an `lvalue`/array argument. This pass
    re-types each payload-buffer function argument to a C++ reference
    (`const T&` for a read-only input, `T&`/`std::array<T,N>&` for a written
    out-param/in-place buffer), re-emits cross-function calls to such functions
    as `emitc.verbatim` text calls (their operand types no longer match the
    reference-typed parameters), and strips HEIR-internal `tensor_ext.*`
    boundary metadata that `mlir-translate --mlir-to-cpp` cannot parse.

    (* example filepath=tests/Conversions/CheddarToEmitC/doctest.mlir *)
  }];

  let dependentDialects = [
    "::mlir::emitc::EmitCDialect",
    "::mlir::func::FuncDialect",
  ];
}

def CheddarExternalizeWeights : Pass<"cheddar-externalize-weights"> {
  let summary = "Externalize large EmitC weight globals to .bin files + a loader.";

  let description = [{
    Runs on EmitC IR (after `--convert-to-emitc`), before
    `mlir-translate --mlir-to-cpp`. The generic C++ translator can only
    serialize *inline* `dense<>` global initializers, not `dense_resource<>`
    blobs (how torch-mlir stores large weight tensors) and would otherwise
    inline huge `dense<>` constants verbatim (gigabytes for big models). This
    mirrors the lattigo emitter's `--data-dir`, but for the emitc path it is a
    real pass rather than emitter string-bashing:

      * For each `emitc.global` whose initializer is a `dense_resource` (any
        size) or a non-splat `dense` with more than `threshold` elements, the
        raw float bytes are written to `<data-dir>/<name>.bin` and the
        initializer is stripped (the global becomes a plain uninitialized
        array).
      * A `__load_constants()` function of `emitc.verbatim` calls is generated
        (`heir_load_f32("data/<name>.bin", reinterpret_cast<float*>(<name>),
        <n>);`) for the harness to call once before inference.

    No-op when `data-dir` is empty.
  }];

  let options = [
    Option<"dataDir", "data-dir", "std::string", "", "Directory to write <name>.bin weight blobs into (empty = disabled)">,
    Option<"threshold", "threshold", "int64_t", "1024", "Inline dense (non-resource) constants with <= this many elements">,
  ];

  let dependentDialects = [
    "::mlir::emitc::EmitCDialect",
    "::mlir::func::FuncDialect",
  ];
}

#endif  // LIB_CONVERSIONS_CHEDDARTOEMITC_CHEDDARTOEMITC_TD_
