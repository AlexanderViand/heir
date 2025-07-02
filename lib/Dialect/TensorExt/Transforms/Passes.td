#ifndef LIB_DIALECT_TENSOREXT_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_TENSOREXT_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

// Pass implementing HECO-style SIMD vectorization for FHE operations.
// This implements automatic SIMD batching by converting extract-operate patterns
// into rotation-based vectorized operations optimized for ciphertext computation.
//
// IMPLEMENTATION DETAILS:
// - Pattern matching for extract-operate sequences
// - Target slot analysis for optimal rotation alignment
// - Heuristic-based rotation insertion to minimize total cost
// - Integration with tensor_ext dialect for rotation operations
//
// REUSABILITY PATTERNS:
// - Modular pattern system allows extension to new operation types
// - Target slot analysis can be customized for different FHE schemes
// - Rotation optimization heuristics can be refined or replaced
// - Can be adapted for different ciphertext slot arrangements
//
// DESIGN CONSIDERATIONS:
// - Focuses on plaintext IR before scheme dialect lowering
// - Relies on subsequent passes for cleanup and optimization
// - Designed for unrolled affine loops and straight-line code
// - Balances vectorization benefits against rotation overhead
def InsertRotate : Pass<"insert-rotate"> {
  let summary = "Vectorize arithmetic FHE operations using HECO-style heuristics";
  let description = [{
  This pass implements the SIMD-vectorization passes from the
  [HECO paper](https://arxiv.org/abs/2202.01649).

  The pass operates by identifying arithmetic operations that can be suitably
  combined into a combination of cyclic rotations and vectorized operations
  on tensors. It further identifies a suitable "slot target" for each operation
  and heuristically aligns the operations to reduce unnecessary rotations.

  This pass by itself does not eliminate any operations, but instead inserts
  well-chosen rotations so that, for well-structured code (like unrolled affine loops),
  a subsequent `--cse` and `--canonicalize` pass will dramatically reduce the IR.
  As such, the pass is designed to be paired with the canonicalization patterns
  in `tensor_ext`, as well as the `collapse-insertion-chains` pass, which
  cleans up remaining insertion and extraction ops after the main simplifications
  are applied.

  Unlike HECO, this pass operates on plaintext types and tensors, along with
  the HEIR-specific `tensor_ext` dialect for its cyclic `rotate` op. It is intended
  to be run before lowering to a scheme dialect like `bgv`.
  }];
  let dependentDialects = ["mlir::heir::tensor_ext::TensorExtDialect"];
}

// TODO(#512): Investigate replacing this pattern with a tensor_ext.combine op
// Pass for optimizing insertion chain patterns into efficient rotations.
// This cleanup pass converts chains of extract/insert operations into single
// rotation operations when the pattern represents a consistent data movement.
//
// IMPLEMENTATION DETAILS:
// - Pattern recognition for complete insertion chains
// - Shift calculation and validation for consistent data movement
// - Chain traversal following tensor value semantics
// - Replacement with equivalent tensor_ext.rotate operations
//
// REUSABILITY PATTERNS:
// - Algorithm can be extended to multi-dimensional tensors
// - Pattern matching framework can handle more complex chain types
// - Validation logic can be adapted for different rotation semantics
// - Can be generalized to other extract/insert optimization patterns
//
// DESIGN CONSIDERATIONS:
// - Requires complete chain coverage for correctness
// - Assumes constant indices (may need prior canonicalization)
// - Limited to 1D tensors for implementation simplicity
// - Designed as cleanup pass for insert-rotate transformations
def CollapseInsertionChains : Pass<"collapse-insertion-chains"> {
  let summary = "Collapse chains of extract/insert ops into rotate ops when possible";
  let description = [{
  This pass is a cleanup pass for `insert-rotate`. That pass sometimes leaves
  behind a chain of insertion operations like this:

  ```mlir
  %extracted = tensor.extract %14[%c5] : tensor<16xi16>
  %inserted = tensor.insert %extracted into %dest[%c0] : tensor<16xi16>
  %extracted_0 = tensor.extract %14[%c6] : tensor<16xi16>
  %inserted_1 = tensor.insert %extracted_0 into %inserted[%c1] : tensor<16xi16>
  %extracted_2 = tensor.extract %14[%c7] : tensor<16xi16>
  %inserted_3 = tensor.insert %extracted_2 into %inserted_1[%c2] : tensor<16xi16>
  ...
  %extracted_28 = tensor.extract %14[%c4] : tensor<16xi16>
  %inserted_29 = tensor.insert %extracted_28 into %inserted_27[%c15] : tensor<16xi16>
  yield %inserted_29 : tensor<16xi16>
  ```

  In many cases, this chain will insert into every index of the `dest` tensor,
  and the extracted values all come from consistently aligned indices of the same
  source tensor. In this case, the chain can be collapsed into a single `rotate`.

  Each index used for insertion or extraction must be constant; this may
  require running `--canonicalize` or `--sccp` before this pass to apply
  folding rules (use `--sccp` if you need to fold constant through control flow).
  }];
  let dependentDialects = ["mlir::heir::tensor_ext::TensorExtDialect"];
}

// Pass implementing logarithmic tensor reduction using rotation operations.
// This transforms linear reduction chains into tree-structured reductions that
// leverage SIMD operations for improved efficiency in FHE computations.
//
// IMPLEMENTATION DETAILS:
// - Pattern matching for complete tensor reductions with associative operations
// - Tree construction using powers-of-2 rotation distances
// - Validation of commutative and associative binary operation requirements
// - Integration with tensor_ext dialect for rotation-based optimizations
//
// REUSABILITY PATTERNS:
// - Algorithm can be extended to different associative operations
// - Tree construction logic can be adapted for various tensor sizes
// - Pattern matching framework can handle more complex reduction patterns
// - Can be generalized to multi-dimensional tensor reductions
//
// DESIGN CONSIDERATIONS:
// - Requires unrolled tensor extractions (typically from loop unrolling)
// - Optimized for power-of-2 tensor sizes for balanced trees
// - Assumes rotation operations are more efficient than individual extractions
// - Designed for FHE schemes with native SIMD rotation support
def RotateAndReduce : Pass<"rotate-and-reduce"> {
  let summary = "Use a logarithmic number of rotations to reduce a tensor.";
  let description = [{
  This pass identifies when a commutative, associative binary operation is used
  to reduce all of the entries of a tensor to a single value, and optimizes the
  operations by using a logarithmic number of reduction operations.

  In particular, this pass identifies an unrolled set of operations of the form
  (the binary ops may come in any order):

  ```mlir
  %0 = tensor.extract %t[0] : tensor<8xi32>
  %1 = tensor.extract %t[1] : tensor<8xi32>
  %2 = tensor.extract %t[2] : tensor<8xi32>
  %3 = tensor.extract %t[3] : tensor<8xi32>
  %4 = tensor.extract %t[4] : tensor<8xi32>
  %5 = tensor.extract %t[5] : tensor<8xi32>
  %6 = tensor.extract %t[6] : tensor<8xi32>
  %7 = tensor.extract %t[7] : tensor<8xi32>
  %8 = arith.addi %0, %1 : i32
  %9 = arith.addi %8, %2 : i32
  %10 = arith.addi %9, %3 : i32
  %11 = arith.addi %10, %4 : i32
  %12 = arith.addi %11, %5 : i32
  %13 = arith.addi %12, %6 : i32
  %14 = arith.addi %13, %7 : i32
  ```

  and replaces it with a logarithmic number of `rotate` and `addi` operations:

  ```mlir
  %0 = tensor_ext.rotate %t, 4 : tensor<8xi32>
  %1 = arith.addi %t, %0 : tensor<8xi32>
  %2 = tensor_ext.rotate %1, 2 : tensor<8xi32>
  %3 = arith.addi %1, %2 : tensor<8xi32>
  %4 = tensor_ext.rotate %3, 1 : tensor<8xi32>
  %5 = arith.addi %3, %4 : tensor<8xi32>
  ```
  }];
  let dependentDialects = ["mlir::heir::tensor_ext::TensorExtDialect"];
}

// Pass implementing the Vos-Vos-Erkin algorithm for efficient permutation networks.
// This converts tensor permutations into optimized shift networks using graph
// coloring to minimize latency in FHE schemes with rotation capabilities.
//
// IMPLEMENTATION DETAILS:
// - Graph coloring algorithm for permutation decomposition
// - Mask generation for element selection in SIMD operations
// - Rotation distance calculation for efficient data movement
// - Integration with tensor_ext dialect for FHE-specific operations
//
// REUSABILITY PATTERNS:
// - Graph coloring approach can be extended to other permutation problems
// - Mask generation logic can be adapted for different SIMD schemes
// - Algorithm can be parameterized for different ciphertext sizes
// - Can be extended to support non-power-of-2 tensor sizes
//
// DESIGN CONSIDERATIONS:
// - Based on published research with theoretical guarantees
// - Optimized for cyclic permutations common in SIMD packing
// - Balances parallel groups against rotation complexity
// - Designed for FHE schemes where rotation cost dominates
def ImplementShiftNetwork : Pass<"implement-shift-network"> {
  let summary = "Implement tensor_ext.convert_layout ops as shift newtorks";

  let description = [{
  This pass converts `tensor_ext.permute` ops into a network of
  `tensor_ext.rotate` ops, aiming to minimize the overall latency of the
  permutation.

  The input IR must have tensors that correspond to plaintexts or
  ciphertexts.

  The method uses graph coloring, an approach based on Vos-Vos-Erkin 2022,
  ["Efficient Circuits for Permuting and Mapping Packed Values Across
   Leveled Homomorphic Ciphertexts"](https://link.springer.com/chapter/10.1007/978-3-031-17140-6_20).

  Example, Figure 3 from the paper above:

  ```mlir
  // Provide an explicit permutation, though an affine_map can also be used.
  #map = dense<[13, 8, 4, 0, 11, 7, 14, 5, 15, 3, 12, 6, 10, 2, 9, 1]> : tensor<16xi64>
  func.func @figure3(%0: tensor<16xi32>) -> tensor<16xi32> {
    %1 = tensor_ext.permute %0 {permutation = #map} : tensor<16xi32>
    return %1 : tensor<16xi32>
  }
  ```

  Then running `--implement-shift-network=ciphertext-size=16` produces
  a shift network composed of plaintext-ciphertext masks (`arith.constant` + `arith.muli`)
  followed by rotations and additions. The Vos-Vos-Erkin method splits the work
  into multiple independent groups that are added together at the end.

  ```mlir
  func.func @figure3(%arg0: tensor<16xi32>) -> tensor<16xi32> {
    %cst = arith.constant dense<[1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]> : tensor<16xi32>
    %0 = arith.muli %arg0, %cst : tensor<16xi32>
    %c1_i32 = arith.constant 1 : i32
    %1 = tensor_ext.rotate %0, %c1_i32 : tensor<16xi32>, i32
    %cst_0 = arith.constant dense<[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<16xi32>
    %2 = arith.muli %arg0, %cst_0 : tensor<16xi32>
    %c2_i32 = arith.constant 2 : i32
    %3 = tensor_ext.rotate %2, %c2_i32 : tensor<16xi32>, i32
    %4 = arith.addi %1, %3 : tensor<16xi32>
    %cst_1 = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]> : tensor<16xi32>
    %5 = arith.muli %arg0, %cst_1 : tensor<16xi32>
    %c4_i32 = arith.constant 4 : i32
    %6 = tensor_ext.rotate %5, %c4_i32 : tensor<16xi32>, i32
    %7 = arith.addi %4, %6 : tensor<16xi32>
    %cst_2 = arith.constant dense<[0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]> : tensor<16xi32>
    %8 = arith.muli %arg0, %cst_2 : tensor<16xi32>
    %c8_i32 = arith.constant 8 : i32
    %9 = tensor_ext.rotate %8, %c8_i32 : tensor<16xi32>, i32
    %10 = arith.addi %7, %9 : tensor<16xi32>
    %cst_3 = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]> : tensor<16xi32>
    %11 = arith.muli %arg0, %cst_3 : tensor<16xi32>
    %c1_i32_4 = arith.constant 1 : i32
    %12 = tensor_ext.rotate %11, %c1_i32_4 : tensor<16xi32>, i32
    %cst_5 = arith.constant dense<[0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0]> : tensor<16xi32>
    %13 = arith.muli %arg0, %cst_5 : tensor<16xi32>
    %c2_i32_6 = arith.constant 2 : i32
    %14 = tensor_ext.rotate %13, %c2_i32_6 : tensor<16xi32>, i32
    %15 = arith.addi %12, %14 : tensor<16xi32>
    %cst_7 = arith.constant dense<[0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0]> : tensor<16xi32>
    %16 = arith.muli %arg0, %cst_7 : tensor<16xi32>
    %c4_i32_8 = arith.constant 4 : i32
    %17 = tensor_ext.rotate %16, %c4_i32_8 : tensor<16xi32>, i32
    %18 = arith.addi %15, %17 : tensor<16xi32>
    %cst_9 = arith.constant dense<[0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]> : tensor<16xi32>
    %19 = arith.muli %arg0, %cst_9 : tensor<16xi32>
    %c8_i32_10 = arith.constant 8 : i32
    %20 = tensor_ext.rotate %19, %c8_i32_10 : tensor<16xi32>, i32
    %21 = arith.addi %18, %20 : tensor<16xi32>
    %22 = arith.addi %10, %21 : tensor<16xi32>
    %cst_11 = arith.constant dense<[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]> : tensor<16xi32>
    %23 = arith.muli %arg0, %cst_11 : tensor<16xi32>
    %c1_i32_12 = arith.constant 1 : i32
    %24 = tensor_ext.rotate %23, %c1_i32_12 : tensor<16xi32>, i32
    %cst_13 = arith.constant dense<[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]> : tensor<16xi32>
    %25 = arith.muli %arg0, %cst_13 : tensor<16xi32>
    %c2_i32_14 = arith.constant 2 : i32
    %26 = tensor_ext.rotate %25, %c2_i32_14 : tensor<16xi32>, i32
    %27 = arith.addi %24, %26 : tensor<16xi32>
    %cst_15 = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]> : tensor<16xi32>
    %28 = arith.muli %arg0, %cst_15 : tensor<16xi32>
    %c4_i32_16 = arith.constant 4 : i32
    %29 = tensor_ext.rotate %28, %c4_i32_16 : tensor<16xi32>, i32
    %30 = arith.addi %27, %29 : tensor<16xi32>
    %cst_17 = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]> : tensor<16xi32>
    %31 = arith.muli %arg0, %cst_17 : tensor<16xi32>
    %c8_i32_18 = arith.constant 8 : i32
    %32 = tensor_ext.rotate %31, %c8_i32_18 : tensor<16xi32>, i32
    %33 = arith.addi %30, %32 : tensor<16xi32>
    %34 = arith.addi %22, %33 : tensor<16xi32>
    return %34 : tensor<16xi32>
  }
  ```
  }];

  let dependentDialects = ["mlir::heir::tensor_ext::TensorExtDialect"];

  // TODO(#4102): reevaluate flag name
  let options = [
    Option<
      "ciphertextSize",
      "ciphertext-size",
      "int",
      /*default=*/"1024",
      "Power of two length of the ciphertexts the data is packed in."
    >
  ];
}

// Pass optimizing layout assignment by eliminating unnecessary conversions.
// This merges convert_layout operations into assign_layout operations to
// directly encode cleartexts into the target layout without intermediate steps.
//
// IMPLEMENTATION DETAILS:
// - Pattern matching for assign_layout followed by convert_layout
// - Single and multiple consumer handling with different strategies
// - Direct layout assignment replacement for optimization
// - Integration with tensor_ext dialect layout management system
//
// REUSABILITY PATTERNS:
// - Pattern-based optimization framework can be extended to other layout ops
// - Multiple consumer handling logic can be adapted for other merge scenarios
// - Can be generalized to other immediate producer-consumer optimizations
// - Direct encoding strategy can be applied to other conversion elimination passes
//
// DESIGN CONSIDERATIONS:
// - Focuses on eliminating redundant layout transformations
// - Optimizes the common pattern of encode-then-convert
// - Handles both single and multiple consumer scenarios efficiently
// - Designed as a cleanup pass for layout propagation artifacts
def FoldConvertLayoutIntoAssignLayout : Pass<"fold-convert-layout-into-assign-layout"> {
  let summary = "Merges tensor_ext.convert_layout ops into preceding tensor_ext.assign_layout ops";
  let description = [{
  A `tensor_ext.assign_layout` op corresponds to an encoding of a cleartext
  into a plaintext or ciphertext. If this is immediately followed by a
  `tensor_ext.convert_layout` op, then one can just change the initial encoding
  to correspond to the result of the conversion.

  If the result of an `assign_layout` has multiple subsequent `convert_layout`
  ops, then they are folded into multiple `assign_layout` ops applied to the
  same cleartext.
  }];
  let dependentDialects = ["mlir::heir::tensor_ext::TensorExtDialect"];
}

#endif  // LIB_DIALECT_TENSOREXT_TRANSFORMS_PASSES_TD_
