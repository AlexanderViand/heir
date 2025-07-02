#ifndef LIB_TRANSFORMS_FULLLOOPUNROLL_FULLLOOPUNROLL_TD_
#define LIB_TRANSFORMS_FULLLOOPUNROLL_FULLLOOPUNROLL_TD_

include "mlir/Pass/PassBase.td"

def FullLoopUnroll : Pass<"full-loop-unroll"> {
  let summary = "Fully unroll all loops";
  let description = [{
  Scans the IR for affine.for loops and unrolls them completely, expanding each loop
  iteration into explicit operations. This transformation is essential for FHE circuit
  generation and optimization pipelines where loop bounds are statically known and
  unrolling enables better constant folding and analysis.

  The pass only operates on affine.for loops with constant bounds and will fail if
  applied to loops with dynamic bounds or very large iteration counts that would
  result in excessive code expansion.

  Developer Notes:
  - Critical preprocessing step for FHE circuit generation and tensor optimizations
  - Enables downstream constant folding and pattern matching optimizations
  - Requires statically analyzable loop bounds (affine.for with constant bounds)
  - Code size consideration: Unrolling can significantly increase IR size
  - Performance trade-off: Eliminates loop overhead but increases compilation time
  - Commonly paired with apply-folders to fold constants created by unrolling
  - Reusable transformation: Core unrolling logic applicable to other loop types
  - Memory impact: Unrolled loops may reveal optimization opportunities for memref operations
  - Pipeline placement: Typically early in optimization pipeline before specialized passes

  Example transformation:
  ```mlir
  // Input: Affine loop with constant bounds
  affine.for %i = 0 to 4 {
    %0 = arith.addi %arg0, %i : index
    // ... operations using %0
  }

  // Output: Fully unrolled operations
  %0 = arith.addi %arg0, 0 : index
  // ... operations using %0
  %1 = arith.addi %arg0, 1 : index
  // ... operations using %1
  %2 = arith.addi %arg0, 2 : index
  // ... operations using %2
  %3 = arith.addi %arg0, 3 : index
  // ... operations using %3
  ```
  }];
}

#endif  // LIB_TRANSFORMS_FULLLOOPUNROLL_FULLLOOPUNROLL_TD_
