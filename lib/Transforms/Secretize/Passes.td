#ifndef LIB_TRANSFORMS_SECRETIZE_PASSES_TD_
#define LIB_TRANSFORMS_SECRETIZE_PASSES_TD_

include "mlir/Pass/PassBase.td"

def Secretize : Pass<"secretize", "ModuleOp"> {
  let summary = "Adds secret argument attributes to entry function";

  let description = [{
    Helper pass that adds a secret.secret attribute argument to each function argument.
    By default, the pass applies to all functions in the module.
    This may be overridden with the option -function=func_name to apply to a single function only.

    This pass is typically the first step in a secretization pipeline. It marks function
    arguments that should be treated as encrypted inputs, preparing them for subsequent
    transformations that will convert the IR to use the Secret dialect.

    Implementation details:
    - Iterates through all functions in the module (or a specific function if specified)
    - Adds `{secret.secret}` attribute to each function argument
    - Does not modify function bodies or return types
    - Uses FuncOp::setArgAttr to attach the secret.secret attribute
    - Handles both public and private functions uniformly
    - Preserves all existing function attributes and structure

    Reusable patterns for developers:
    1. Module-level iteration: Shows how to iterate through all functions in a module
    2. Selective function targeting: Demonstrates filtering functions by name
    3. Argument attribute manipulation: Pattern for adding attributes to function arguments
    4. Attribute creation: Example of creating StringAttr attributes for annotations

    Extension opportunities:
    - Could be extended to support more sophisticated secret marking strategies
    - Pattern can be adapted for other forms of function argument annotation
    - Could be extended to mark return values as secret as well
    - Framework can be reused for other dialect-specific function annotations
  }];

  let dependentDialects = [
    "mlir::heir::secret::SecretDialect",
    "mlir::func::FuncDialect"
  ];

  let options = [
    Option<"function", "function", "std::string", "\"\"", "function to add secret annotations to">
  ];
}

def WrapGeneric : Pass<"wrap-generic", "ModuleOp"> {
  let summary = "Wraps regions using secret args in secret.generic bodies";

  let description = [{
    This pass converts functions (`func.func`) with `{secret.secret}` annotated arguments
    to use `!secret.secret<...>` types and wraps the function body in a `secret.generic` region.
    The output type is also converted to `!secret.secret<...>`.

    This pass is the second step in the secretization pipeline, following the `secretize` pass.
    It establishes the Secret dialect structure by creating the `secret.generic` wrapper that
    encapsulates the computation to be performed on encrypted data.

    Implementation details:
    - Scans all functions for those with `{secret.secret}` argument attributes
    - Converts argument types from T to `!secret.secret<T>` using SecretType::get()
    - Wraps the entire function body in a single `secret.generic` operation
    - Creates block arguments for the generic region matching the unwrapped types
    - Updates return statements to use `secret.yield`
    - Converts function return type to `!secret.secret<T>`
    - Handles multiple return values by wrapping each in secret types
    - Uses IRMapping to maintain value correspondence between original and new bodies
    - Preserves function visibility and other attributes

    Reusable patterns for developers:
    1. Type conversion: Shows systematic conversion of function signatures with secret types
    2. Region wrapping: Demonstrates how to wrap existing computation in new dialect operations
    3. Block argument creation: Pattern for creating new block arguments with converted types
    4. Operation cloning: Example of moving operations between regions while updating types
    5. Return value handling: Shows conversion of func.return to dialect-specific yield operations
    6. IRMapping usage: Best practice for maintaining value mappings during transformations

    Extension opportunities:
    - Could be extended to handle more complex function signatures (variadic, etc.)
    - Pattern can be adapted for other secret-like wrapper dialects
    - Could support partial secretization (some args secret, some not)
    - Framework can be extended to handle nested function calls
    - Could be enhanced to preserve debug information and source locations

    Example input:
    ```mlir
      func.func @main(%arg0: i32 {secret.secret}) -> i32 {
        %0 = arith.constant 100 : i32
        %1 = arith.addi %0, %arg0 : i32
        return %1 : i32
      }
    ```

    Output:
    ```mlir
      func.func @main(%arg0: !secret.secret<i32>) -> !secret.secret<i32> {
        %0 = secret.generic(%arg0 : !secret.secret<i32>) {
        ^bb0(%arg1: i32):
          %1 = arith.constant 100 : i32
          %2 = arith.addi %1, %arg1 : i32
          secret.yield %2 : i32
        } -> !secret.secret<i32>
        return %0 : !secret.secret<i32>
      }
    ```
  }];

  let dependentDialects = [
    "mlir::heir::secret::SecretDialect",
    "mlir::func::FuncDialect"
  ];
}

#endif  // LIB_TRANSFORMS_SECRETIZE_PASSES_TD_
