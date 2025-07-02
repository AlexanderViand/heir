#ifndef LIB_DIALECT_CGGI_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_CGGI_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def SetDefaultParameters : Pass<"cggi-set-default-parameters"> {
  let summary = "Set default parameters for CGGI ops";
  let description = [{
  // DEVELOPER NOTE: This is a utility pass for testing and development that provides
  // hard-coded default parameters for CGGI operations. It serves as a temporary solution
  // while proper parameter generation mechanisms are being developed.
  //
  // Implementation details:
  // - Traverses all CGGI operations in the module
  // - Applies hard-coded cggi_params attributes to each operation
  // - Overrides any existing parameters (useful for testing different parameter sets)
  // - Parameters are defined in SetDefaultParameters.cpp implementation
  //
  // Reusability considerations:
  // - The pass structure can be reused for other dialects needing default parameters
  // - Parameter values should be updated as cryptographic research advances
  // - Consider making parameters configurable via pass options for better reusability
  // - This approach can be adapted for other testing scenarios requiring parameter injection
  This pass adds default parameters to all CGGI ops as `cggi_params` named
  attributes, overriding any existing attribute set with that name.

  This pass is primarily for testing purposes, and as a parameter provider
  before a proper parameter selection mechanism is added. This pass should not
  be used in production.

  The specific parameters are hard-coded in
  `lib/Dialect/CGGI/Transforms/SetDefaultParameters.cpp`.
  }];
  let dependentDialects = [
    "mlir::heir::cggi::CGGIDialect",
    "mlir::heir::mod_arith::ModArithDialect",
  ];
}

def DecomposeOperations : Pass<"cggi-decompose-operations"> {
  let summary = "Expands CGGI operations into LWE operations and programmable bootstraps";
  let description = [{
    // DEVELOPER NOTE: This pass implements the core decomposition strategy for CGGI operations
    // by expanding high-level boolean gates into the fundamental LWE operations required for
    // TFHE-style computation. The implementation uses a two-level expansion approach.
    //
    // Implementation details:
    // - Level 1: High-level ops (XOR, LUT2, LUT3) -> cggi.lut_lincomb operations
    // - Level 2: cggi.lut_lincomb -> LWE scalar ops + programmable bootstrap (PBS)
    // - Uses pattern-based rewriting with operation-specific decomposition rules
    // - Supports configurable expansion depth via expand-lincomb option
    //
    // Key algorithmic insight:
    // - Multi-input boolean functions can be encoded as single-input lookup tables
    //   by treating input bits as coefficients in a weighted sum (e.g., 4*c + 2*b + a)
    // - This leverages PBS efficiency where cost is dominated by table lookup, not input size
    //
    // Reusability considerations:
    // - The pattern-based expansion framework can be extended to new CGGI operations
    // - The linear combination encoding strategy applies to any boolean function
    // - The two-level approach allows backends to choose their preferred abstraction level
    This pass expands high level CGGI operations (e.g. LUT2, XOR, etc.).

    If the option `expand-lincomb` is set, the expansion will continue into the
    component LWE scalar operations and a programmable bootstrap operation.
    Otherwise, the expansion will be stop at the `cggi.lut_lincomb` level. By
    default, `expand-lincomb` is true.

    For example, a LUT3 operation is composed of three LWE ciphertext inputs $c,
    b, a$ (in MSB to LSB ordering) which must be combined via the linear
    combination $4 * c + 2 * b + a$ before being fed into a programmable
    bootstrap defined by the lookup table.

    This pass supports XOR, LUT2, LUT3, and LutLincomb operations.
  }];
  let dependentDialects = ["mlir::heir::cggi::CGGIDialect"];

  let options = [
    Option<"expandLincomb", "expand-lincomb", "bool",
           /*default=*/"true", "Expand lincomb operations to the PBS and scalar level">
  ];
}

def BooleanVectorizer : Pass<"cggi-boolean-vectorize"> {
  let summary = "Group different logic gates with the packed API";
  let description = [{
    // DEVELOPER NOTE: This pass implements vectorization for boolean operations by grouping
    // independent gates into batch operations. Unlike traditional vectorization that combines
    // identical operations, this pass can combine heterogeneous boolean gates.
    //
    // Implementation details:
    // - Analyzes independence of boolean gate operations within the function
    // - Groups compatible operations based on parallelism constraints
    // - Generates packed operation calls with gate type vectors and operand tensors
    // - Maintains semantic equivalence while reducing function call overhead
    //
    // Key algorithmic insights:
    // - Independence analysis ensures correctness of parallel execution
    // - Heterogeneous grouping maximizes batching opportunities
    // - Backend API mapping translates IR structure to efficient library calls
    // - Parallelism parameter allows tuning for hardware constraints
    //
    // Backend integration:
    // - Specifically designed for TFHE-rs FPT (Fast Packed Transform) API
    // - Maps to packed_gates function taking gate specifications and operand vectors
    // - Enables hardware acceleration through batch processing
    // - Reduces function call overhead in generated code
    //
    // Reusability considerations:
    // - Framework can be adapted for other boolean logic backends with packed APIs
    // - Independence analysis generalizes to other operation types
    // - Vectorization strategy applicable to any batched computation model
    // - Parallelism control enables backend-specific optimization
    //
    // Performance benefits:
    // - Reduced function call overhead through batching
    // - Better memory locality from grouped operations
    // - Hardware acceleration opportunities through packed operations
    // - Backend optimization potential for grouped computations
    This pass groups independent logic gates into a single call of the packed
    operations. Pass is based on the straight-line-vectorizer, but is fundamentally different.
    This pass combines any type of boolean gates and is not restricted to combining the same type of gate operand.

    Pass is intended for the `FPT` tfhe-rs API, where `packed_gates` function get a
    the boolean gates are passed as a string vector and a left and right vector of ciphertexts.
    Each boolean gates specified in `gates` is then applied element wise.

    ```
    let outputs_ct = fpga_key.packed_gates(&gates, &ref_to_ct_lefts, &ref_to_ct_rights);
    ```
  }];

  let options = [
    Option<"parallelism", "parallelism", "int",
           /*default=*/"0", "Parallelism factor for batching. 0 is infinite parallelism">
  ];

  let dependentDialects = [
    "tensor::TensorDialect",
  ];
}

#endif  // LIB_DIALECT_CGGI_TRANSFORMS_PASSES_TD_
