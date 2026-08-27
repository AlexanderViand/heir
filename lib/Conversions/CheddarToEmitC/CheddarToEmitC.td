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
    as structured `emitc.call_opaque` operations (their operand types no longer
    match the reference-typed parameters), and strips HEIR-internal `tensor_ext.*`
    boundary metadata that `mlir-translate --mlir-to-cpp` cannot parse.

    (* example filepath=tests/Conversions/CheddarToEmitC/doctest.mlir *)
  }];

  // Also list the dialects that the preceding `--convert-to-emitc` restricts to
  // via `filter-dialects=cheddar,arith,scf`: that pass errors if a
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

def CheddarEmitCEntryInterface
    : Pass<"cheddar-emitc-entry-interface", "mlir::ModuleOp"> {
  let summary = "Generate a C++ entry-point interface and split file output";
  let description = [{
    Materializes the logical entry-point roles recorded by
    `--add-client-interface` as a compact C++ API. The pass creates separate
    `emitc.file` regions for the generated header and source and keeps the
    lowered helper functions private to the source file.

    Run after `--cheddar-emitc-boundary` and
    `--reconcile-unrealized-casts`.
  }];
  let dependentDialects = [
    "::mlir::emitc::EmitCDialect",
    "::mlir::func::FuncDialect",
  ];
  let options = [
    Option<"entryFunction", "entry-function", "std::string", "\"\"",
           "Entry function to expose when the module contains more than one">
  ];
}

#endif  // LIB_CONVERSIONS_CHEDDARTOEMITC_CHEDDARTOEMITC_TD_
