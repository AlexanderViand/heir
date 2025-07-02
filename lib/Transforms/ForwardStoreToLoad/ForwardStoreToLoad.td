#ifndef LIB_TRANSFORMS_FORWARDSTORETOLOAD_FORWARDSTORETOLOAD_TD_
#define LIB_TRANSFORMS_FORWARDSTORETOLOAD_FORWARDSTORETOLOAD_TD_

include "mlir/Pass/PassBase.td"

def ForwardStoreToLoad : Pass<"forward-store-to-load"> {
  let summary = "Forward stores to loads within a single block";
  let description = [{
    This pass is a simplified version of mem2reg and similar passes.
    It analyzes an operation, finding all basic blocks within that op
    that have memrefs whose stores can be forwarded to loads.

    Does not support complex control flow within a block, nor ops
    with arbitrary subregions.

    Developer Notes:
    - Simplified mem2reg implementation focused on single-block scenarios
    - Essential for scalar replacement and memory optimization in FHE pipelines
    - Eliminates redundant memory operations by direct value forwarding
    - Limitation: Only operates within single basic blocks, no inter-block analysis
    - Works with affine.store/affine.load and memref.store/memref.load operations
    - Performance optimization: Reduces memory traffic by eliminating store-load pairs
    - Commonly used after loop unrolling when memory access patterns become clear
    - Reusable component: Core forwarding logic can be extended for more complex scenarios
    - Prerequisites: Requires static memory access analysis (affine expressions preferred)
    - Pipeline placement: Typically after unrolling but before more complex optimizations

    Example transformation:
    ```mlir
    // Input: Store followed by load of same location
    %alloc = memref.alloc() : memref<1xi32>
    affine.store %value, %alloc[0] : memref<1xi32>
    %loaded = affine.load %alloc[0] : memref<1xi32>
    // ... use %loaded

    // Output: Direct value forwarding
    %alloc = memref.alloc() : memref<1xi32>
    // store eliminated
    // load eliminated, replaced with direct use of %value
    // ... use %value
    ```
  }];
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::memref::MemRefDialect"
  ];
}

#endif  // LIB_TRANSFORMS_FORWARDSTORETOLOAD_FORWARDSTORETOLOAD_TD_
