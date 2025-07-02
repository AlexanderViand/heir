#ifndef LIB_DIALECT_MGMT_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_MGMT_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def AnnotateMgmt : Pass<"annotate-mgmt"> {
  let summary = "Annotate MgmtAttr for secret SSA values in the IR";
  let description = [{
  // DEVELOPER NOTE: This pass implements three critical dataflow analyses for FHE compilation:
  // secretness analysis, ciphertext level tracking, and dimension analysis. The implementation
  // uses standard MLIR dataflow analysis infrastructure to propagate management information.
  //
  // Implementation details:
  // - Secretness Analysis: Determines which values are encrypted vs plaintext using lattice-based analysis
  // - Level Analysis: Tracks multiplication depth to determine ciphertext levels (crucial for parameter sizing)
  // - Dimension Analysis: Tracks vector dimensions for SIMD-packed ciphertexts
  // - Annotation: Stores analysis results as mgmt.mgmt attributes on operations
  //
  // Key algorithmic insights:
  // - Uses forward dataflow analysis with lattice meet operations
  // - Secretness: SECRET ⊔ PUBLIC = SECRET (secret values propagate)
  // - Levels: max(input_levels) for addition, max(input_levels) + 1 for multiplication
  // - Dimensions: element-wise max for compatible dimensions, unknown for incompatible
  //
  // Reusability considerations:
  // - Analysis framework can be extended for other ciphertext properties
  // - Base level option allows adaptation to different FHE schemes (BGV starts at level 0, some BFV at level 1)
  // - Annotation format is extensible for additional management metadata
  // - Conservative analysis ensures correctness at the cost of some optimization opportunities
  //
  // Integration points:
  // - Parameter generation passes read these annotations to size schemes appropriately
  // - Optimization passes use level information for relinearization placement
  // - Backend passes use secretness for encryption/plaintext operation selection
  This pass runs the secretness/level/dimension analysis and annotates the IR with the results,
  saving it into each op's attribute dictionary as mgmt.mgmt : <analysis results>
  }];
  let dependentDialects = ["mlir::heir::mgmt::MgmtDialect"];
  let options = [
    Option<"baseLevel", "base-level", "int",
           /*default=*/"0", "Level to start counting from (used by B/FV)">,
  ];
}

#endif  // LIB_DIALECT_MGMT_TRANSFORMS_PASSES_TD_
