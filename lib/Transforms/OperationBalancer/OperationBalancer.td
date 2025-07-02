#ifndef LIB_TRANSFORMS_OPERATIONBALANCER_OPERATIONBALANCER_TD_
#define LIB_TRANSFORMS_OPERATIONBALANCER_OPERATIONBALANCER_TD_

include "mlir/Pass/PassBase.td"

def OperationBalancer : Pass<"operation-balancer"> {
  let summary = "This pass balances addition and multiplication operations.";

  let dependentDialects = [
    "arith::ArithDialect"
  ];

  let description = [{
  // DEVELOPER NOTE: This pass implements arithmetic expression tree balancing to optimize
  // FHE circuit depth and expose parallelization opportunities. The implementation uses
  // graph analysis to identify and restructure associative operation chains.
  //
  // Implementation details:
  // - Walks through secret.generic operations containing arithmetic expressions
  // - Builds dependency graphs for addition and multiplication chains
  // - Applies depth-first search to identify balancing opportunities
  // - Reconstructs expression trees with minimal depth while preserving semantics
  //
  // Key algorithmic insights:
  // - Focuses on associative operations (addition, multiplication) that can be reordered
  // - Handles intermediate values used multiple times as separate subtrees (conservative but correct)
  // - Minimizes local tree depth rather than global optimization (tractable and effective)
  // - Preserves operation dependencies and semantic equivalence
  //
  // Balancing strategy:
  // - Unbalanced: ((a + b) + c) + d has depth 3, sequential execution
  // - Balanced: (a + b) + (c + d) has depth 2, parallel execution possible
  // - Critical for FHE where depth determines parameter requirements and performance
  //
  // Reusability considerations:
  // - Graph analysis framework can be extended to other algebraic structures
  // - Tree balancing algorithms applicable to any associative operations
  // - Secret.generic restriction can be relaxed or adapted for other encrypted computation models
  // - Conservative handling of shared intermediate values ensures correctness across different optimization contexts
  //
  // FHE-specific benefits:
  // - Reduced multiplication depth enables smaller FHE parameters (exponential savings)
  // - Parallel structure maps well to SIMD-capable FHE backends
  // - Better cache locality from structured computation patterns
  // - Enables more effective relinearization placement in subsequent passes
  //
  // Research foundation:
  // Based on expression tree balancing techniques from compiler optimization literature,
  // specifically adapted for FHE constraints as described in EVA compiler research.
  This pass examines a tree or graph of add and multiplication operations and
  balances them to minimize the depth of the tree. This exposes better parallelization
  and reducing the multiplication depth can decrease the parameters used in FHE,
  which improves performance. This pass is not necessarily optimal, as there may
  be intermediate computations that this pass does not optimally minimize the depth for.

  The algorithm is to analyze a graph of addition operations and do a depth-first
  search for the operands (from the last computed values in the graph). If there
  are intermediate computations that are used more than once, then the pass
  treats that computation as its own tree to balance instead of trying to minimize
  the global depth of the tree.

  This pass only runs on addition and multiplication operations on the arithmetic
  dialect that are encapsulated inside a secret.generic.

  This pass was inspired by section 2.6 of ['EVA Improved: Compiler and Extension
  Library for CKKS' by Chowdhary et al](https://eprint.iacr.org/2021/1505).
  }];
}

#endif  // LIB_TRANSFORMS_OPERATIONBALANCER_OPERATIONBALANCER_TD_
