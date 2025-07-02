#ifndef LIB_TRANSFORMS_MEMREFTOARITH_MEMREFTOARITH_TD_
#define LIB_TRANSFORMS_MEMREFTOARITH_MEMREFTOARITH_TD_

include "mlir/Pass/PassBase.td"

def MemrefGlobalReplacePass : Pass<"memref-global-replace", "mlir::ModuleOp"> {
  let summary = "MemrefGlobalReplacePass forwards global memrefs accessors "
  "to arithmetic values";

  let description = [{
    This pass forwards constant global MemRef values to referencing affine
    loads. This pass requires that the MemRef global values are initialized as
    constants and that the affine load access indices are constants (i.e. not
    variadic). Unroll affine loops prior to running this pass.

     MemRef removal is required to remove any memory allocations from the input
     model (for example, TensorFlow models contain global memory holding model
     weights) to support FHE transpilation.

    Implementation details:
    - Walks through all affine.load operations in the module
    - Tracks the source of each load back to memref.get_global operations
    - Verifies that global memrefs are constant-initialized with DenseElementsAttr
    - Evaluates affine access patterns using AffineValueMap for index resolution
    - Replaces loads with direct arith.constant operations containing the accessed values
    - Removes unused memref.get_global and memref.global operations
    - Uses affine expression analysis to handle complex indexing patterns
    - Maintains type consistency between loaded values and generated constants

    Developer implementation patterns:
    1. Global analysis: Shows module-level analysis for tracking global constants
    2. Affine expression evaluation: Demonstrates static evaluation of affine access patterns
    3. Constant extraction: Pattern for extracting values from DenseElementsAttr
    4. Dead code elimination: Framework for removing unused global operations
    5. Type preservation: Shows maintaining type consistency during transformation
    6. Symbol table usage: Demonstrates proper symbol resolution for global references

    Reusable components:
    - Affine index evaluator: Can be used for other constant folding transformations
    - Global constant tracker: Framework applicable to other global memory optimizations
    - Dense attribute accessor: Pattern for extracting typed values from dense attributes
    - Dead operation removal: Systematic approach to cleaning up unused globals

    Extension opportunities:
    - Could be extended to handle non-dense constant initializers
    - Pattern can support partial constant folding (some indices remain dynamic)
    - Could be enhanced to handle more complex affine maps
    - Framework can support global variable dependency analysis

     Input

     ```
     module {
       memref.global "private" constant @__constant_8xi16 : memref<2x4xi16> = dense<[[-10, 20, 3, 4], [5, 6, 7, 8]]>
       func.func @main() -> i16 {
         %c1 = arith.constant 1 : index
         %c2 = arith.constant 2 : index
         %0 = memref.get_global @__constant_8xi16 : memref<2x4xi16>
         %1 = affine.load %0[%c1, %c1 + %c2] : memref<2x4xi16>
         return %1 : i16
       }
     }
     ```

     Output

     ```
     module {
       func.func @main() -> i16 {
         %c1 = arith.constant 1 : index
         %c2 = arith.constant 2 : index
         %c8_i16 = arith.constant 8 : i16
         return %c8_i16 : i16
       }
     }
     ```

  }];

  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
  ];
}

def ExpandCopyPass : Pass<"expand-copy", "mlir::ModuleOp"> {
  let summary = "Expands memref.copy ops to explicit affine loads and stores";

  let description = [{
    This pass removes memref copy operations by expanding them to affine loads
    and stores. This pass introduces affine loops over the dimensions of the
    MemRef, so must be run prior to any affine loop unrolling in a pipeline.

    Input

    ```
    module {
      func.func @memref_copy() {
        %alloc = memref.alloc() : memref<2x3xi32>
        %alloc_0 = memref.alloc() : memref<2x3xi32>
        memref.copy %alloc, %alloc_0 : memref<1x1xi32> to memref<1x1xi32>
      }
    }
    ```

    Output

    ```
    module {
      func.func @memref_copy() {
        %alloc = memref.alloc() : memref<2x3xi32>
        %alloc_0 = memref.alloc() : memref<2x3xi32>
        affine.for %arg0 = 0 to 2 {
          affine.for %arg1 = 0 to 3 {
            %1 = affine.load %alloc[%arg0, %arg1] : memref<2x3xi32>
            affine.store %1, %alloc_0[%arg0, %arg1] : memref<2x3xi32>
          }
        }
      }
    }
    ```

    When `--disable-affine-loop=true` is set, then the output becomes
    ```
    module {
      func.func @memref_copy() {
        %alloc = memref.alloc() : memref<2x3xi32>
        %alloc_0 = memref.alloc() : memref<2x3xi32>
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %0 = affine.load %alloc[%c0, %c0] : memref<2x3xi32>
        affine.store %0, %alloc_0[%c0, %c0] : memref<2x3xi32>
        %1 = affine.load %alloc[%c0, %c1] : memref<2x3xi32>
        affine.store %1, %alloc_0[%c0, %c1] : memref<2x3xi32>
        %2 = affine.load %alloc[%c0, %c2] : memref<2x3xi32>
        affine.store %2, %alloc_0[%c0, %c2] : memref<2x3xi32>
        [...]
      }
    }
    ```
  }];

  let options = [
    Option<"disableAffineLoop", "disable-affine-loop", "bool", /*default=*/"false",
           "Use this to control to disable using affine loops">,
  ];

  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
  ];
}

def UnrollAndForwardPass : Pass<"unroll-and-forward", "func::FuncOp"> {
  let summary = "Loop unrolls and forwards stores to loads.";

  let description = [{
    // DEVELOPER NOTE: This pass implements aggressive memory elimination for FHE compilation
    // by combining loop unrolling with sophisticated store-to-load forwarding. The goal is
    // to eliminate all memory operations in favor of register-based computation.
    //
    // Implementation details:
    // - Iterative processing: unroll loops one at a time, then forward within unrolled code
    // - Sophisticated alias analysis to track memref values through subviews and renames
    // - Static index analysis to enable precise store-load matching
    // - Conservative handling of function arguments to preserve interface semantics
    // - Integration with global memref handling (delegates to memref-global-replace)
    //
    // Key algorithmic components:
    // - Dataflow analysis to track store-load relationships
    // - Alias tracking through memref operations (subview, cast, etc.)
    // - Index analysis for static affine expressions
    // - Value forwarding with dependency preservation
    //
    // Store-load forwarding algorithm:
    // 1. For each load with static index, find the defining memref allocation
    // 2. Collect all stores to the same memref at the same index
    // 3. Identify the last store that dominates the load
    // 4. Replace load with the stored value, preserving all dependencies
    // 5. Handle function arguments specially (forward through renames only)
    //
    // Reusability considerations:
    // - Unrolling infrastructure can be reused for other loop-based optimizations
    // - Store-load forwarding framework applicable to other memory elimination scenarios
    // - Alias analysis components useful for other memref transformations
    // - Index analysis reusable for other static analysis passes
    //
    // FHE-specific benefits:
    // - Eliminates memory allocation (cannot encrypt arbitrary memory layouts)
    // - Creates register-only computation suitable for encrypted domains
    // - Enables better analysis and optimization of arithmetic operations
    // - Reduces complexity for subsequent FHE-specific transformations
    //
    // Limitations and design choices:
    // - Requires static loop bounds and array access patterns (common in ML workloads)
    // - Processes functions individually (modular approach for large programs)
    // - Excludes global memrefs (handled by specialized pass for different analysis requirements)
    This pass processes the first function in a given module, and, starting from
    the first loop, iteratively does the following:

    1. Fully unroll the loop.
    2. Scan for load ops. For each load op with a statically-inferrable access
    index:
      1. Backtrack to the original memref alloc
      2. Find all store ops at the corresponding index (possibly transitively
      through renames/subviews of the underlying alloc).
      3. Find the last store that occurs and forward it to the load.
      4. If the original memref is an input memref, then forward through any
      renames to make the target load load directly from the argument memref
      (instead of any subviews, say)
    3. Apply the same logic to any remaining loads not inside any for loop.

    This pass requires that tensors are lowered to memref, and only supports
    affine loops with affine.load/store ops.

    Memrefs that result from memref.get_global ops are excluded from
    forwarding, even if they are loaded with a static index, and are instead
    handled by memref-global-replace, which should be run after this pass.

  }];

  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
  ];
}

def ExtractLoopBodyPass : Pass<"extract-loop-body", "mlir::ModuleOp"> {
  let summary = "Extracts logic of a loop bodies into functions.";

  let description = [{
    // DEVELOPER NOTE: This pass implements function extraction from loop bodies to create
    // modular, reusable computational kernels. The transformation separates iteration logic
    // from computation, enabling better optimization and analysis of individual components.
    //
    // Implementation details:
    // - Analyzes loop structure to identify extraction candidates based on size thresholds
    // - Extracts computational logic while preserving load-compute-store patterns
    // - Generates function signatures based on loaded values (inputs) and stored values (outputs)
    // - Maintains semantic equivalence by replacing loop body with function call
    // - Uses systematic naming convention for generated functions
    //
    // Extraction criteria:
    // - Loop iteration count meets minimum threshold (avoids overhead for small loops)
    // - Loop body operation count meets minimum threshold (ensures meaningful extraction)
    // - Loop body follows expected pattern: loads → computation → single store
    // - No complex control flow within loop body (maintains simplicity)
    //
    // Function generation strategy:
    // - Function parameters: Values loaded within the loop body
    // - Function return: Value that would be stored by the loop body
    // - Function body: All computational operations between loads and store
    // - Constants: Moved into extracted function to maintain self-containment
    //
    // Reusability considerations:
    // - Framework can be extended to other loop extraction patterns
    // - Threshold parameters allow tuning for different optimization goals
    // - Generated functions are amenable to independent optimization
    // - Modular structure enables potential function reuse across different contexts
    //
    // Benefits for analysis and optimization:
    // - Function-level optimization: Each kernel can be optimized independently
    // - Simplified loop analysis: Loops become simple iteration with function calls
    // - Better testing: Individual computational kernels can be unit tested
    // - Profiling granularity: Performance analysis at the kernel level
    // - Vectorization opportunities: Extracted functions may be good vectorization candidates
    //
    // Use cases:
    // - Neural network layer implementations with repeated element-wise operations
    // - Image processing with per-pixel or per-block computations
    // - Mathematical kernels with regular computation patterns
    // - Any scenario where loop body complexity warrants modularization
    This pass extracts logic in the inner body of for loops into functions.

    This pass requires that tensors are lowered to memref. It expects that a
    loop body contains a number of affine.load statements used as inputs to the
    extracted function, and a single affine.store used as the extracted
    function's output.

    Input

    ```
    module {
      func.func @loop_body() {
        %c-128_i8 = arith.constant -128 : i8
        %c127_i8 = arith.constant 127 : i8
        %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<25x20x8xi8>
        affine.for %arg1 = 0 to 25 {
          affine.for %arg2 = 0 to 20 {
            affine.for %arg3 = 0 to 8 {
              %98 = affine.load %alloc_6[%arg1, %arg2, %arg3] : memref<25x20x8xi8>
              %99 = arith.cmpi slt, %arg0, %c-128_i8 : i8
              %100 = arith.select %99, %c-128_i8, %arg0 : i8
              %101 = arith.cmpi sgt, %arg0, %c127_i8 : i8
              %102 = arith.select %101, %c127_i8, %100 : i8
              affine.store %102, %alloc_7[%arg1, %arg2, %arg3] : memref<25x20x8xi8>
            }
          }
        }
      }
    }
    ```

    Output

    ```
    module {
      func.func @loop_body() {
        %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<25x20x8xi8>
        affine.for %arg1 = 0 to 25 {
          affine.for %arg2 = 0 to 20 {
            affine.for %arg3 = 0 to 8 {
              %98 = affine.load %alloc_6[%arg1, %arg2, %arg3] : memref<25x20x8xi8>
              %102 = func.call @__for_loop(%98) : (i8) -> i8
              affine.store %102, %alloc_7[%arg1, %arg2, %arg3] : memref<25x20x8xi8>
            }
          }
        }
      }
      func.func private @__for_loop(%arg0: i8) -> i8 {
        %c-128_i8 = arith.constant -128 : i8
        %c127_i8 = arith.constant 127 : i8
        %99 = arith.cmpi slt, %arg0, %c-128_i8 : i8
        %100 = arith.select %99, %c-128_i8, %arg0 : i8
        %101 = arith.cmpi sgt, %arg0, %c127_i8 : i8
        %102 = arith.select %101, %c127_i8, %100 : i8
        return %102 : i8
      }
    }
    ```
  }];

  let options = [
    Option<"minimumLoopSize", "min-loop-size", "unsigned", /*default=*/"4",
           "Use this to control the minimum loop size to apply this pass">,
    Option<"minimumBodySize", "min-body-size", "unsigned", /*default=*/"4",
           "Use this to control the minimum loop body size to apply this pass">,
  ];

  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
  ];
}

#endif  // LIB_TRANSFORMS_MEMREFTOARITH_MEMREFTOARITH_TD_
