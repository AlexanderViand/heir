#ifndef LIB_TRANSFORMS_MODARITHTOARITHVEIR_MODARITHTOARITHVEIR_TD_
#define LIB_TRANSFORMS_MODARITHTOARITHVEIR_MODARITHTOARITHVEIR_TD_

include "mlir/Pass/PassBase.td"

def ModArithToArithVeir : Pass<"mod-arith-to-arith-veir", "ModuleOp"> {
  let summary = "Lower `mod_arith` to `arith` using the verified VEIR optimizer";
  let description = [{
    This pass lowers the `mod_arith` dialect to the `arith` dialect by shelling
    out to [VEIR](https://github.com/BelfortLabs/veir), a formally verified,
    Lean-based MLIR-compatible optimizer. VEIR's `mod-arith-to-arith` pass and
    the correctness proofs of its rewrites are written in Lean 4, so the
    lowered IR is semantics-preserving by construction.

    The pass prints the module in the generic op form to a temporary file,
    invokes `veir-opt` on it with the pipeline given by the `pipeline` option,
    parses the result back, and replaces the contents of the module with the
    lowered IR. Like HEIR's native `--mod-arith-to-arith` lowering, the result
    may contain `builtin.unrealized_conversion_cast` ops at `func.func`
    boundaries; run `--reconcile-unrealized-casts` afterwards to clean up any
    cancelling pairs.

    The `veir-opt` binary is not built by HEIR; it is built out-of-band with
    `lake build` in the VEIR repository (producing `.lake/build/bin/veir-opt`).
    The binary is located, in order of precedence, via:

    1. the `veir-opt-path` pass option,
    2. the `HEIR_VEIR_OPT_PATH` environment variable,
    3. looking up `veir-opt` on the `PATH`.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
  ];
  let options = [
    Option<"veirOptPath", "veir-opt-path", "std::string", /*default=*/"\"\"",
           "Path to the veir-opt binary. If empty, falls back to the "
           "HEIR_VEIR_OPT_PATH environment variable, then to looking up "
           "veir-opt on the PATH.">,
    Option<"pipeline", "pipeline", "std::string",
           /*default=*/"\"mod-arith-to-arith,reconcile-cast\"",
           "Comma-separated list of VEIR passes to run, passed to veir-opt "
           "as a single -p=<pipeline> argument.">,
  ];
}

#endif  // LIB_TRANSFORMS_MODARITHTOARITHVEIR_MODARITHTOARITHVEIR_TD_
