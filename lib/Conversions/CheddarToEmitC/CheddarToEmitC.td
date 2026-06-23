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

#endif  // LIB_CONVERSIONS_CHEDDARTOEMITC_CHEDDARTOEMITC_TD_
