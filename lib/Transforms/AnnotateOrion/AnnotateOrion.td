#ifndef LIB_TRANSFORMS_ANNOTATEORION_ANNOTATEORION_TD_
#define LIB_TRANSFORMS_ANNOTATEORION_ANNOTATEORION_TD_

include "mlir/Pass/PassBase.td"

def AnnotateOrion : Pass<"annotate-orion"> {
  let summary = "Annotate Orion ops with implementation style and coarse level cost";
  let description = [{
    This pass selects a generic implementation style for Orion compound ops and
    annotates each op with the coarse intrinsic level cost expected by later
    management and parameter-selection passes.

    Backend lowerings that consume Orion ops are expected to honor the
    annotated implementation style or fail loudly if they cannot realize it.
  }];

  let options = [
    Option<"linearTransformImplStyle", "linear-transform-impl-style",
           "std::string", /*default=*/"\"diagonal-bsgs\"",
           "Implementation style to request for orion.linear_transform">,
    Option<"chebyshevImplStyle", "chebyshev-impl-style", "std::string",
           /*default=*/"\"bsgs\"",
           "Implementation style to request for orion.chebyshev">,
  ];
}

#endif  // LIB_TRANSFORMS_ANNOTATEORION_ANNOTATEORION_TD_
