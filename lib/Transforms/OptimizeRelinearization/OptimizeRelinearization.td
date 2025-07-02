#ifndef LIB_TRANSFORMS_OPTIMIZE_RELINEARIZATION_TD_
#define LIB_TRANSFORMS_OPTIMIZE_RELINEARIZATION_TD_

include "mlir/Pass/PassBase.td"

def OptimizeRelinearization : Pass <"optimize-relinearization"> {
    let summary = "Optimize placement of relinearization ops";
    let description = [{
        // DEVELOPER NOTE: This pass implements global optimization of relinearization placement
        // using Integer Linear Programming (ILP) to minimize noise accumulation and operation count.
        // Relinearization is expensive both computationally and in terms of noise growth, making
        // optimal placement critical for FHE performance.
        //
        // Implementation details:
        // - Formulates relinearization placement as an ILP optimization problem
        // - Uses binary variables for each potential relinearization site
        // - Incorporates ciphertext degree tracking and key basis constraints
        // - Solves per-function to maintain tractable problem sizes
        // - Generates optimized IR with relinearizations moved to optimal positions
        //
        // ILP model components:
        // - Variables: Binary indicators for relinearization placement, integer degree variables
        // - Constraints: Degree compatibility, return value linearization, rotation requirements
        // - Objective: Minimize total number of relinearization operations
        // - Solution: Optimal relinearization schedule satisfying all constraints
        //
        // Key algorithmic insights:
        // - Defers relinearization until mathematically necessary (operation requirements)
        // - Exploits the fact that additions work on any ciphertext degree
        // - Minimizes noise accumulation by reducing early relinearizations
        // - Enables better parallelization by reducing false dependencies
        //
        // Mathematical foundation:
        // - Based on dataflow analysis principles adapted for FHE-specific constraints
        // - Uses constraint programming techniques from compiler optimization
        // - Incorporates domain-specific knowledge about FHE operation requirements
        //
        // Reusability considerations:
        // - ILP framework can be extended for other FHE optimization problems
        // - Constraint modeling approach applies to other resource-constrained optimizations
        // - Function-level analysis can be composed for whole-program optimization
        // - Backend-agnostic formulation enables optimization across different FHE libraries
        //
        // Performance implications:
        // - Reduces expensive relinearization operations (significant runtime savings)
        // - Minimizes noise growth, potentially avoiding bootstrapping
        // - Improves parallelization opportunities by removing unnecessary dependencies
        // - Enables smaller parameter selection due to reduced noise accumulation
        This pass defers relinearization ops as late as possible in the IR.
        This is more efficient in cases where multiplication operations are followed by
        additions, such as in a dot product. Because relinearization also adds error,
        deferring it can reduce the need for bootstrapping.

        In this pass, we use an integer linear program to determine the optimal
        relinearization strategy. It solves an ILP for each `func` op in the IR.

        The assumptions of this pass include:

        - All return values of functions must be linearized.
        - All ciphertext arguments to an op must have the same key basis
        - Rotation op inputs must have be linearized.

        For an ILP model specification, see the
        [docs at the HEIR website](https://heir.dev/docs/design/relinearization_ilp/).
        The model is an adaptation of the ILP described in
        [a blog post by Jeremy Kun](https://www.jeremykun.com/2023/11/15/mlir-a-global-optimization-and-dataflow-analysis).
    }];

    let dependentDialects = ["mlir::heir::mgmt::MgmtDialect"];

  let options = [
    // When using this option, one should generally write the IR to disk and
    // run `--optimize-relinearization` as the first or only pass, otherwise the
    // ILP model construction may fail due to having non-unique variable names
    Option<"useLocBasedVariableNames",
           "use-loc-based-variable-names",
           "bool",
           /*default=*/"false",
           "When true, the ILP uses op source locations in variable names, "
           "which can help debug ILP model bugs.">,
    Option<"allowMixedDegreeOperands",
           "allow-mixed-degree-operands",
           "bool",
           /*default=*/"true",
           "When true, allow ops to have mixed-degree ciphertexts as inputs, e.g., "
           "adding two ciphertexts with different key bases; this is supported by "
           "many FHE backends, like OpenFHE and Lattigo">,
  ];
}

#endif  // LIB_TRANSFORMS_OPTIMIZE_RELINEARIZATION_TD_
