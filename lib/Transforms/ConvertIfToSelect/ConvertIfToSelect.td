#ifndef LIB_TRANSFORMS_CONVERTIFTOSELECT_CONVERTIFTOSELECT_TD_
#define LIB_TRANSFORMS_CONVERTIFTOSELECT_CONVERTIFTOSELECT_TD_

include "mlir/Pass/PassBase.td"

def ConvertIfToSelect : Pass<"convert-if-to-select"> {
  let summary = "Convert scf.if operations on secret conditions to arith.select operations.";
  let description = [{
  Converts `scf.if` operations with secret-dependent conditions to `arith.select` operations,
  eliminating branching that depends on secret values. This transformation is the final step
  in creating data-oblivious programs where execution flow does not reveal information about
  secret inputs.

  This pass completes the data-oblivious transformation pipeline that typically begins with
  passes like `convert-secret-while-to-static-for`, `convert-secret-for-to-static-for`, and
  `convert-secret-extract-to-static-extract`, which introduce conditional operations that this
  pass then converts to select operations.

  Developer Notes:
  - Final transformation in data-oblivious compilation pipeline
  - Eliminates all secret-dependent control flow branching
  - Security critical: Ensures constant-time execution regardless of secret values
  - Works by evaluating both branches and selecting result using arith.select
  - Handles multi-value yields by creating multiple select operations
  - Preserves SSA form and value dependencies throughout transformation
  - Reusable pattern: Core logic applicable to other conditional selection scenarios
  - Performance consideration: Both branches execute, increasing computational cost
  - Enables FHE circuit generation by removing all data-dependent branching

  Example transformation:
  ```mlir
  // Input: Secret-dependent conditional
  %result = scf.if %secret_condition -> (i32) {
    %a = arith.addi %x, %y : i32
    scf.yield %a : i32
  } else {
    %b = arith.muli %x, %y : i32
    scf.yield %b : i32
  }

  // Output: Select-based computation
  %a = arith.addi %x, %y : i32
  %b = arith.muli %x, %y : i32
  %result = arith.select %secret_condition, %a, %b : i32
  ```
  }];
  let dependentDialects = [
    "mlir::scf::SCFDialect",
    "mlir::arith::ArithDialect"
  ];
}

#endif  // LIB_TRANSFORMS_CONVERTIFTOSELECT_CONVERTIFTOSELECT_TD_
