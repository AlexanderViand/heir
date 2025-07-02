#ifndef LIB_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_MODARITHTOARITH_TD_
#define LIB_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_MODARITHTOARITH_TD_

include "lib/Utils/DRR/Utils.td"
include "lib/Dialect/ModArith/IR/ModArithOps.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/IR/PatternBase.td"
include "mlir/Pass/PassBase.td"

/// **Developer Documentation: ModArithToArith Pass**
///
/// This pass implements the reverse conversion from modular arithmetic operations
/// back to standard arithmetic operations, enabling integration with standard
/// MLIR optimization infrastructure and backend targeting.
///
/// **Implementation Details:**
/// - Converts modular arithmetic operations to standard arithmetic equivalents
/// - Uses declarative rewrite rules (DRR) for pattern-based conversion
/// - Preserves modular arithmetic semantics using standard operations
/// - Handles specialized operations like conditional subtraction
///
/// **Pattern-Based Approach:**
/// - Leverages MLIR's DRR framework for conversion patterns
/// - Extensible pattern system for new modular operations
/// - Specific patterns for complex operations (e.g., SubIfGE)
/// - Direct mapping for common arithmetic operations
///
/// **Integration Benefits:**
/// - Enables standard MLIR arithmetic optimizations
/// - Compatible with existing backend targeting infrastructure
/// - Allows analysis passes that work on standard arithmetic
/// - Provides flexibility in compilation pipeline design
///
/// **Semantic Preservation:**
/// - Maintains modular arithmetic overflow behavior
/// - Implements conditional operations using standard control flow
/// - Preserves mathematical correctness of modular operations
/// - Compatible with standard arithmetic analysis passes
///
/// **Extension Points:**
/// - Add patterns for new modular arithmetic operations
/// - Implement specialized lowering strategies for specific backends
/// - Integrate with custom arithmetic optimization passes
def ModArithToArith : Pass<"mod-arith-to-arith", "ModuleOp"> {
  let summary = "Lower `mod_arith` to standard `arith`.";

  let description = [{
    This pass lowers the `mod_arith` dialect to their `arith` equivalents.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::tensor::TensorDialect",
    "mlir::heir::mod_arith::ModArithDialect",
  ];
}

// Using DRR to generate the lowering patterns for specific operations

defvar DefGE = ConstantEnumCase<Arith_CmpIPredicateAttr, "uge">;

def ConvertSubIfGE : Pattern<
  (ModArith_SubIfGEOp $x, $y),
  [
    (Arith_SubIOp:$subOp $x, $y, DefOverflow),
    (Arith_CmpIOp:$cmpOp DefGE, $x, $y),
    (SelectOp $cmpOp, $subOp, $x)
  ]
>;

#endif  // LIB_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_MODARITHTOARITH_TD_
