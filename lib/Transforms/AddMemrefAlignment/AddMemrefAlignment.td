#ifndef LIB_TRANSFORMS_ADDMEMREFALIGNMENT_ADDMEMREFALIGNMENT_TD_
#define LIB_TRANSFORMS_ADDMEMREFALIGNMENT_ADDMEMREFALIGNMENT_TD_

include "mlir/Pass/PassBase.td"

def AddMemrefAlignmentPass : Pass<"add-memref-alignment", "mlir::ModuleOp"> {
  let summary = "Add alignment attributes to memref.alloc operations";
  let description = [{
    This pass adds alignment attributes to memref.alloc operations that don't
    already have them. This ensures that the LLVM lowering produces properly
    aligned memory allocations, which is required for the memrefCopy runtime
    function to work correctly without segfaults.

    The pass adds 'alignment = 64 : i64' to all memref.alloc operations that
    don't already specify an alignment attribute.
  }];

  let dependentDialects = [
    "mlir::memref::MemRefDialect",
  ];
}

#endif  // LIB_TRANSFORMS_ADDMEMREFALIGNMENT_ADDMEMREFALIGNMENT_TD_
