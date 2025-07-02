#ifndef LIB_TRANSFORMS_STRAIGHTLINEVECTORIZER_STRAIGHTLINEVECTORIZER_TD_
#define LIB_TRANSFORMS_STRAIGHTLINEVECTORIZER_STRAIGHTLINEVECTORIZER_TD_

include "mlir/Pass/PassBase.td"

// Pass for vectorizing straight-line programs (no control flow).
// This implements basic vectorization by converting scalar operations to
// vector/tensor operations where beneficial for SIMD execution.
//
// IMPLEMENTATION DETAILS:
// - Operates only within straight-line regions (ignores control flow)
// - Dialect filtering allows targeted vectorization
// - Simple pattern-based approach for operation conversion
// - Preserves program semantics while exploiting data parallelism
//
// REUSABILITY PATTERNS:
// - Dialect-agnostic framework can be extended to new operations
// - Pattern-based design allows easy addition of vectorization rules
// - Can be specialized for different target architectures
// - Works as preprocessing step for more sophisticated vectorizers
//
// DESIGN CONSIDERATIONS:
// - Limited to straight-line code for simplicity and correctness
// - Does not perform complex dependency analysis
// - Assumes target backend can efficiently execute vector operations
// - Designed as building block for more complex vectorization strategies
def StraightLineVectorizer : Pass<"straight-line-vectorize"> {
  let summary = "A vectorizer for straight line programs.";
  let description = [{
  This pass ignores control flow and only vectorizes straight-line programs
  within a given region.
  }];

  let dependentDialects = [
    "tensor::TensorDialect",
  ];

  let options = [
    Option<"dialect", "dialect", "std::string", "", "Use this to restrict the "
      "dialect whose ops should be vectorized.">,
  ];
}

#endif  // LIB_TRANSFORMS_STRAIGHTLINEVECTORIZER_STRAIGHTLINEVECTORIZER_TD_
