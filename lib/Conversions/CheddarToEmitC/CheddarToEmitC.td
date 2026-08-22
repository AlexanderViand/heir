#ifndef LIB_CONVERSIONS_CHEDDARTOEMITC_CHEDDARTOEMITC_TD_
#define LIB_CONVERSIONS_CHEDDARTOEMITC_CHEDDARTOEMITC_TD_

include "mlir/Pass/PassBase.td"

def CheddarEmitCBoundary : Pass<"cheddar-emitc-boundary", "::mlir::ModuleOp"> {
  let summary = "Re-type CHEDDAR payload buffer arguments as C++ references";

  let description = [{
    Runs after `--convert-to-emitc`. A function cannot carry an `emitc.lvalue`
    argument, so each payload buffer argument is re-typed to a C++ reference
    (`const T&` if only read, `T&` if written), calls to such functions are
    re-emitted as `emitc.call_opaque`, the standard headers the emitted C++
    uses are included, and HEIR-internal `tensor_ext.*` argument attributes
    are dropped.

    (* example filepath=tests/Conversions/CheddarToEmitC/doctest.mlir *)
  }];

  // `--convert-to-emitc=filter-dialects=...` requires the listed dialects to be
  // loaded; this pass always follows it in the pipeline, so load them here.
  let dependentDialects = [
    "::mlir::emitc::EmitCDialect",
    "::mlir::func::FuncDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::heir::cheddar::CheddarDialect",
  ];
}

def CheddarEmitCEntryInterface
    : Pass<"cheddar-emitc-entry-interface", "mlir::ModuleOp"> {
  let summary = "Generate a C++ entry-point interface and split file output";
  let description = [{
    Wraps the facades `--cheddar-build-entry-interface` generated in a
    compact C++ API: public functions over `std::tuple` aggregates of the
    entry's cleartext and encrypted values. The pass creates separate
    `emitc.file` regions for the generated header and source and keeps the
    lowered functions private to the source file.

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
