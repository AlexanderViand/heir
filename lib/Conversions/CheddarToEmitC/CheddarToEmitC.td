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

  // Also list the dialects that the preceding `--convert-to-emitc` restricts to
  // via `filter-dialects=cheddar,arith,scf,memref`: that pass errors if a
  // filtered dialect is not *loaded*, and a program without loops/allocs would
  // not otherwise load scf/memref. Declaring them here (this pass always runs
  // right after convert-to-emitc) forces them loaded at pass-manager init.
  let dependentDialects = [
    "::mlir::emitc::EmitCDialect",
    "::mlir::func::FuncDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::heir::cheddar::CheddarDialect",
  ];
}

def CheddarExternalizeWeights : Pass<"cheddar-externalize-weights"> {
  let summary = "Wire externalized weight globals to a __load_constants() loader.";

  let description = [{
    Runs on EmitC IR (after `--convert-to-emitc`), before
    `mlir-translate --mlir-to-cpp`. The weight blobs themselves are written by
    the upstream `externalize-constants` pass (run before bufferization);
    `--convert-to-emitc` lowers the resulting `preprocessing.load_resource`
    ops to uninitialized `emitc.global`s tagged with their runtime blob path.
    This pass then:

      * Collects the tagged globals into a generated `__load_constants()`
        function of `emitc.verbatim` calls (`heir_load_f32("<path>",
        reinterpret_cast<float*>(<name>), <n>);`, or `heir_load_f64`/`double`
        for f64 globals) for the harness to call once before inference.
      * Strips the initializer from any *large splat* float global (which
        externalize-constants deliberately leaves inline): heir-translate
        expands array initializers element-by-element, so a splat global
        (e.g. a 512x65536 zero-padded ciphertext-width buffer) would balloon
        the emitted C++ by hundreds of MB. An all-zero splat then relies on
        C++ static zero-initialization; a non-zero splat is filled by a loop
        in `__load_constants()`.
  }];

  let options = [
    Option<"threshold", "threshold", "int64_t", "1024", "Keep splat globals with <= this many elements inline">,
  ];

  let dependentDialects = [
    "::mlir::emitc::EmitCDialect",
    "::mlir::func::FuncDialect",
  ];
}

#endif  // LIB_CONVERSIONS_CHEDDARTOEMITC_CHEDDARTOEMITC_TD_
