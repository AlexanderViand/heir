#ifndef LIB_DIALECT_LWE_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_LWE_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"


def AddDebugPort : Pass<"lwe-add-debug-port"> {
  let summary = "Add debug port to (R)LWE encrypted functions";
  let description = [{
  // DEVELOPER NOTE: This debugging pass enables runtime inspection of encrypted computations
  // by instrumenting functions with debug hooks. The implementation modifies function signatures
  // and inserts external function calls that can decrypt and analyze ciphertext values.
  //
  // Implementation details:
  // - Walks through all functions containing LWE operations
  // - Modifies function signatures to include secret key parameters
  // - Inserts calls to external debug functions after each homomorphic operation
  // - Preserves original computation flow while adding inspection capabilities
  //
  // Key design decisions:
  // - Uses external function declarations rather than inline debugging to maintain modularity
  // - Adds secret key as first parameter to enable decryption in debug functions
  // - Follows naming convention "__heir_debug" for easy identification and linking
  // - Maintains operation ordering to preserve debugging context
  //
  // Reusability considerations:
  // - The pass structure can be adapted for other encrypted computation dialects
  // - Debug function interface can be extended for different analysis needs
  // - Entry function filtering allows selective instrumentation
  // - External function approach enables custom debug implementations per backend
  //
  // Security note: This pass is for development only - debug capabilities expose
  // encrypted data and should never be used in production deployments.
  This pass adds debug ports to the specified function in the IR. The debug ports
  are prefixed with "__heir_debug" and are invoked after each homomorphic operation in the
  function. The debug ports are declarations and user should provide functions with
  the same name in their code.

  For example, if the function is called "foo", the secret key is added to its
  arguments, and the debug port is called after each homomorphic operation:
  ```mlir
  // declaration of external debug function
  func.func private @__heir_debug(%sk : !sk, %ct : !ct)

  // secret key added as function arg
  func.func @foo(%sk : !sk, ...) {
    %ct = lwe.radd ...
    // invoke external debug function
    __heir_debug(%sk, %ct)
    %ct1 = lwe.rmul ...
    __heir_debug(%sk, %ct1)
    ...
  }
  ```
  }];
  let dependentDialects = ["mlir::heir::lwe::LWEDialect"];
  let options = [
    Option<"entryFunction", "entry-function", "std::string",
           /*default=*/"", "Default entry function "
           "name of entry function.">,
  ];
}

#endif  // LIB_DIALECT_LWE_TRANSFORMS_PASSES_TD_
