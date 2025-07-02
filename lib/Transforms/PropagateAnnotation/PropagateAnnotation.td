#ifndef LIB_TRANSFORMS_ANNOTATEMODULE_ANNOTATEMODULE_TD_
#define LIB_TRANSFORMS_ANNOTATEMODULE_ANNOTATEMODULE_TD_

include "mlir/Pass/PassBase.td"

// Pass for dataflow propagation of attributes throughout the IR.
// This implements configurable attribute propagation to maintain metadata,
// analysis results, or optimization hints as they flow through computations.
//
// IMPLEMENTATION DETAILS:
// - Configurable attribute name selection for targeted propagation
// - Bidirectional propagation support (forward and reverse)
// - Non-overwriting policy to preserve existing attribute values
// - Operation-level attribute attachment and propagation
//
// REUSABILITY PATTERNS:
// - Attribute name parameterization allows reuse for different metadata types
// - Direction control enables different propagation strategies
// - Can be extended to support more sophisticated propagation rules
// - Framework can be adapted for conditional or filtered propagation
//
// DESIGN CONSIDERATIONS:
// - Preserves existing attributes to avoid unintended overwrites
// - Supports both forward and backward dataflow propagation
// - Designed for maintaining analysis results across transformations
// - Enables attribute-driven optimization passes
def PropagateAnnotation : Pass<"propagate-annotation"> {
  let summary = "Propagate annotation from operation to subsequent operations";
  let description = [{
    This pass propagates the attribute from one operation to subsequent operations
    if these operations does not have the attribute already.

    Example: with `--propagate-annotation=attr-name=test.attr`

    ```mlir
    func.func @foo(%arg0: i16 {test.attr = 1}) -> i16 {
      %0 = arith.muli %arg0, %arg0 : i16
      %1 = mgmt.relinearize %0 : i16
      return %1 : i16
    }
    ```

    the above IR becomes

    ```mlir
    func.func @foo(%arg0: i16 {test.attr = 1 : i64}) -> i16 {
      %0 = arith.muli %arg0, %arg0 {test.attr = 1 : i64} : i16
      %1 = mgmt.relinearize %0 {test.attr = 1 : i64} : i16
      return {test.attr = 1 : i64} %1 : i16
    }
    ```
  }];

  let options = [
    Option<"attrName", "attr-name", "std::string", /*default=*/"\"\"",
           "The attribute name to propagate with.">,
    Option<"reverse", "reverse", "bool", /*default=*/"false",
           "Whether to propagate in reverse">,
  ];
}

#endif  // LIB_TRANSFORMS_ANNOTATEMODULE_ANNOTATEMODULE_TD_
