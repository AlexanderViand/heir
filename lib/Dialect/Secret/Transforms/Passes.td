#ifndef LIB_DIALECT_SECRET_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_SECRET_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def SecretForgetSecrets : Pass<"secret-forget-secrets"> {
  let summary = "Convert secret types to standard types";
  let description = [{
    Drop the `secret<...>` type from the IR, replacing it with the contained
    type and the corresponding cleartext computation.

    This pass effectively reverses secretization by removing all Secret dialect constructs
    and converting encrypted computation back to cleartext computation. It's primarily used
    for testing, validation, and creating reference implementations.

    `secret.cast` ops are replaced with freshly alloc'ed memrefs that extract
    individual bits of the input type or reshape them if possible.

    Implementation details:
    - Removes `!secret.secret<T>` type wrappers, converting them back to `T`
    - Unwraps `secret.generic` operations, inlining their body operations into parent region
    - Removes `secret.conceal` and `secret.reveal` operations (identity transformations)
    - Updates function signatures to use cleartext types via type conversion
    - Converts `secret.yield` operations to appropriate terminator operations
    - Handles function calls by updating call signatures and return types
    - Uses TypeConverter framework for systematic type transformation
    - Handles complex nested structures and maintains operation ordering

    Developer implementation patterns:
    1. Type conversion framework: Uses ConversionTarget and TypeConverter for systematic lowering
    2. Operation inlining: Shows how to inline region operations while maintaining dominance
    3. Terminator handling: Demonstrates conversion of dialect-specific terminators
    4. Function signature updates: Pattern for converting function types during lowering
    5. Cascading conversions: Shows how type changes propagate through the IR
    6. Pattern-based rewriting: Uses ConversionPattern for structured transformation

    Reusable components:
    - SecretTypeConverter: Can be adapted for other secret-like type unwrapping
    - Generic operation inlining: Pattern reusable for other dialect region inlining
    - Function signature conversion: Framework applicable to other dialect lowerings

    Extension opportunities:
    - Could be enhanced to handle more complex secret operations
    - Pattern can be extended for partial lowering (some ops remain secret)
    - Could be augmented with verification of cleartext/secret equivalence
    - Framework can support custom unwrapping strategies per operation type

    Use cases:
    - Testing: Generate cleartext reference implementations
    - Validation: Verify correctness of secret computation logic
    - Debugging: Isolate issues in secret dialect vs. underlying computation
    - Benchmarking: Compare cleartext vs. encrypted performance
  }];
  let dependentDialects = [
    "mlir::heir::secret::SecretDialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect",
  ];
}

def SecretDistributeGeneric : Pass<"secret-distribute-generic"> {
  let summary = "Distribute `generic` ops through their bodies.";
  let description = [{
    Converts `generic` ops whose region contains many ops into smaller
    sequences of generic ops whose regions contain a single op, dropping the
    `generic` part from any resulting `generic` ops that have no
    `secret.secret` inputs. If the op has associated regions, and the operands
    are not secret, then the generic is distributed recursively through the
    op's regions as well.

    This pass is intended to be used as part of a front-end pipeline, where a
    program that operates on a secret type annotates the input to a region as
    `secret`, and then wraps the contents of the region in a single large
    `secret.generic`, then uses this pass to simplify it.

    The `distribute-through` option allows one to specify a comma-separated
    list of op names (e.g., `distribute-thorugh="affine.for,scf.if"`), which
    limits the distribution to only pass through those ops. If unset, all ops
    are distributed through when possible.

    Implementation details:
    - Iterates through `secret.generic` operations in the IR using OpRewritePattern
    - For each operation in the generic body, creates a new individual `secret.generic`
    - Chains the results between consecutive generics maintaining SSA form
    - Handles control flow by distributing through regions (affine.for, scf.if, etc.)
    - Removes generic wrapper if all operands become cleartext (via CollapseSecretlessGeneric)
    - Constants are hoisted outside generic operations for efficiency
    - Uses IRMapping to track value correspondence during transformation
    - Handles complex dependency chains and maintains operation ordering

    Developer implementation patterns:
    1. Operation distribution: Shows systematic breaking of compound operations
    2. Region traversal: Demonstrates recursive processing of nested regions
    3. Dependency tracking: Pattern for maintaining SSA dependencies during splitting
    4. Conditional transformation: Shows operation-specific distribution decisions
    5. Generic operation creation: Framework for creating new generic wrappers
    6. Pattern matching: Uses operation name filtering for selective distribution

    Reusable components:
    - Generic distribution logic: Adaptable for other compound operation breaking
    - Region recursion framework: Pattern for nested region transformations
    - Operation filtering: Mechanism reusable for selective transformations
    - Value chain management: Framework for maintaining complex value dependencies

    Extension opportunities:
    - Could be enhanced with cost models for distribution decisions
    - Pattern can support custom distribution strategies per operation type
    - Could be extended with analysis to determine optimal distribution points
    - Framework can support partial distribution based on usage patterns

    Benefits:
    - Enables fine-grained analysis and optimization of individual operations
    - Prepares IR for backend-specific transformations
    - Simplifies subsequent passes by working on smaller units
    - Allows for better parallelization opportunities
  }];
  let dependentDialects = ["mlir::heir::secret::SecretDialect"];

  let options = [
    ListOption<"opsToDistribute", "distribute-through", "std::string",
               "comma-separated list of ops that should be distributed through">,
  ];
}

def SecretCaptureGenericAmbientScope : Pass<"secret-capture-generic-ambient-scope"> {
  let summary = "Capture the ambient scope used in a secret.generic";
  let description = [{
    For each value used in the body of a `secret.generic` op, which is defined
    in the ambient scope outside the `generic`, add it to the argument list of
    the `generic`.

    This pass makes implicit dependencies explicit by ensuring that all values used
    within a `secret.generic` operation are passed as arguments rather than captured
    from the surrounding scope. This is essential for proper IR structure and enables
    subsequent transformations that rely on explicit argument passing.

    Implementation details:
    - Walks through all `secret.generic` operations in the IR
    - Analyzes the generic body to find uses of values defined outside the generic
    - For each external value used:
      * Adds it as a new argument to the generic operation
      * Creates a corresponding block argument in the generic's entry block
      * Updates all uses within the generic to reference the new block argument
    - Handles nested regions correctly to avoid infinite loops
    - Maintains correct ordering of arguments and block arguments
    - Reusable pattern: This demonstrates scope analysis and argument lifting,
      which can be adapted for other transformations that need to make captures explicit

    Benefits:
    - Enables function extraction from generic bodies
    - Simplifies analysis by making all dependencies explicit
    - Prepares IR for transformations that require well-formed argument lists
    - Improves IR quality by eliminating implicit captures
  }];
  let dependentDialects = ["mlir::heir::secret::SecretDialect"];
}

def SecretMergeAdjacentGenerics : Pass<"secret-merge-adjacent-generics"> {
  let summary = "Merge two adjacent generics into a single generic";
  let description = [{
    This pass merges two immediately sequential generics into a single
    generic. Useful as a sub-operation in some passes, and extracted into
    its own pass for testing purposes.
  }];
  let dependentDialects = ["mlir::heir::secret::SecretDialect"];
}

def SecretGenericAbsorbConstants : Pass<"secret-generic-absorb-constants"> {
  let summary = "Copy constants into a secret.generic body";
  let description = [{
    For each constant value used in the body of a `secret.generic` op, which is
    defined in the ambient scope outside the `generic`, add it's definition into
    the `generic` body.
  }];
  let dependentDialects = ["mlir::heir::secret::SecretDialect"];
}

def SecretGenericAbsorbDealloc : Pass<"secret-generic-absorb-dealloc"> {
  let summary = "Copy deallocs of internal memrefs into a secret.generic body";
  let description = [{
    For each memref allocated and used only within a body of a `secret.generic`
    op, add it's dealloc of the memref into its `generic` body.
  }];
  let dependentDialects = ["mlir::heir::secret::SecretDialect"];
}

def SecretExtractGenericBody : Pass<"secret-extract-generic-body"> {
  let summary = "Extract the bodies of all generic ops into functions";
  let description = [{
    This pass extracts the body of all generic ops into functions, and
    replaces the generic bodies with call ops. Used as a sub-operation in
    some passes, and extracted into its own pass for testing purposes.

    This pass works best when `--secret-generic-absorb-constants` is run
    before it so that the extracted function contains any constants used
    in the generic op's body.
  }];
  let dependentDialects = ["mlir::heir::secret::SecretDialect"];
}

def SecretAddDebugPort : Pass<"secret-add-debug-port"> {
  let summary = "Add debug port to secret-arithmetic ops";
  let description = [{
  This pass adds debug ports to secret-arithmetic ops in the IR, namely operations
  wrapped by secret.generic. The debug ports are prefixed with "__heir_debug" and
  are invoked after each operation in the generic body. The debug ports are
  declarations and user should provide functions with the same name in their code.

  For example, if the function is called "foo", the debug port is called after
  each homomorphic operation:
  ```mlir
  // declaration of external debug function
  func.func private @__heir_debug_tensor_8xi16_(tensor<8xi16>)

  func.func @foo(...) {
    secret.generic {
      %0 = arith.addi ...
      // invoke external debug function
      __heir_debug_tensor_8xi16_(%0)
      %1 = arith.muli ...
      __heir_debug_tensor_8xi16_(%1)
    }
  }
  ```
  }];
}

def SecretImportExecutionResult : Pass<"secret-import-execution-result"> {
  let summary = "Annotate execution result to secret-arithmetic ops";
  let description = [{
  When the execution result of each op is known by `secret-add-debug-port` pass,
  the result could be imported back to the IR.

  This pass adds a new attribute `secret.execution_result` to the secret-arithmetic ops.

  This is useful when users want to compare the precision of the result between
  the plaintext and the ciphertext (especially the CKKS case).

  For example, if you have a `trace.log` that is generated by plaintext backend with
  `--secret-add-debug-port` where the result is printed out like

  ```
  1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0
  2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0
  ```

  Each _line_ corresponds to one SSA value in the IR. You can then import the result
  back to the IR by using `--secret-import-execution-result=file-name=trace.log`.

  ```mlir
  func.func @foo(...) {
    secret.generic {
      %0 = arith.addi ... {secret.execution_result = [1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0]}
      %1 = arith.muli ... {secret.execution_result = [2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0]}
    }
  }
  ```
  }];

  let options = [
    Option<"fileName", "file-name", "std::string",
           /*default=*/"", "file name of the execution result">
  ];
}


#endif  // LIB_DIALECT_SECRET_TRANSFORMS_PASSES_TD_
