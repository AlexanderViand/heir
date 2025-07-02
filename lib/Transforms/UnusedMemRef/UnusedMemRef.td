#ifndef LIB_TRANSFORMS_UNUSEDMEMREF_UNUSEDMEMREF_TD_
#define LIB_TRANSFORMS_UNUSEDMEMREF_UNUSEDMEMREF_TD_

include "mlir/Pass/PassBase.td"

// Pass for dead code elimination of unused memory allocations.
// This cleanup pass removes locally allocated memrefs that have no uses,
// typically after optimization passes that eliminate memory references.
//
// IMPLEMENTATION DETAILS:
// - Function-scoped analysis for local allocation identification
// - Usage analysis to determine if allocations have any references
// - Dead code elimination for completely unused allocations
// - Integration with memory optimization pipeline as cleanup pass
//
// REUSABILITY PATTERNS:
// - Dead code elimination framework can be extended to other allocation types
// - Usage analysis can be adapted for more sophisticated aliasing scenarios
// - Can be generalized to other resource management optimizations
// - Scope-limited analysis can be applied to other local resource cleanup
//
// DESIGN CONSIDERATIONS:
// - Limited to locally allocated memrefs for safety
// - Conservative approach avoids removing potentially aliased memory
// - Designed as post-optimization cleanup for store-to-load forwarding
// - Balances optimization benefits against compilation safety
def RemoveUnusedMemRef : Pass<"remove-unused-memref", "func::FuncOp"> {
  let summary = "Cleanup any unused memrefs";
  let description = [{
    Scan the IR for unused memrefs and remove them.

    This pass looks for locally allocated memrefs that are never used and
    deletes them. This pass can be used as a cleanup pass from other IR
    simplifications that forward stores to loads.
  }];
}

#endif  // LIB_TRANSFORMS_UNUSEDMEMREF_UNUSEDMEMREF_TD_
