#ifndef LIB_TRANSFORMS_ANNOTATEMODULE_ANNOTATEMODULE_TD_
#define LIB_TRANSFORMS_ANNOTATEMODULE_ANNOTATEMODULE_TD_

include "mlir/Pass/PassBase.td"

def AnnotateModule : Pass<"annotate-module"> {
  let summary = "Annotate ModuleOp with Scheme And/Or Backend";
  let description = [{
    This pass annotates the module with a scheme and/or backend.

    This pass should be called before all lowering to enable lowering
    to the desired scheme and backend.

    Available scheme:
      - `bgv`
      - `ckks`
      - `bfv`
      - `cggi`

    Available backend:
      - `openfhe`
      - `lattigo`

    Example: `--annotate-module="backend=openfhe scheme=ckks"`

    ```
    module attributes {backend.openfhe, scheme.ckks}
      ...
    }
    ```

    Developer Notes:
    - Uses ModuleAttributes.h helper functions for consistent attribute naming
    - Scheme and backend selections are independent - can specify one or both
    - Enables conditional lowering and backend-specific optimizations in subsequent passes
    - Module attributes guide target-specific code generation and optimization decisions
    - Reusable pattern: Easy to extend with new schemes/backends by adding cases
      and corresponding moduleSet* functions
    - No validation of scheme/backend compatibility - handled by later passes
  }];

  let options = [
    Option<"scheme", "scheme", "std::string", /*default=*/"\"\"",
           "The scheme to annotate the module with.">,
    Option<"backend", "backend", "std::string", /*default=*/"\"\"",
            "The backend to annotate the module with.">,
  ];
}

#endif  // LIB_TRANSFORMS_ANNOTATEMODULE_ANNOTATEMODULE_TD_
