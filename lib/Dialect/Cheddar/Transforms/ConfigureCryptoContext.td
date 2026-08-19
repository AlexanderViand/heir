#ifndef LIB_DIALECT_CHEDDAR_TRANSFORMS_CONFIGURECRYPTOCONTEXT_TD_
#define LIB_DIALECT_CHEDDAR_TRANSFORMS_CONFIGURECRYPTOCONTEXT_TD_

include "mlir/Pass/PassBase.td"

def CheddarConfigureCryptoContext : Pass<"cheddar-configure-crypto-context"> {
  let summary = "Configure CHEDDAR crypto context from scheme parameters.";

  let description = [{
    This pass reads the `ckks.schemeParam` module attribute and re-exposes the
    CHEDDAR-relevant parameters (`cheddar.logN`, `cheddar.logDefaultScale`,
    `cheddar.Q`, `cheddar.P`) as module attributes, then removes the CKKS
    module attributes.

    It also generates a `<entry>__configure` function that builds the CHEDDAR
    `Parameter` from the modulus chain, creates the `Context` and
    `UserInterface`, and prepares a rotation key for every distinct rotation
    distance used by the program (`cheddar.hrot` / `cheddar.hrot_add`). The
    `prepare-rotation-keys-at-use-levels` option instead prepares every
    distance at each operation level where it is used. The context and
    user_interface are handed back to the caller through owning
    out-params (`std::shared_ptr<Context<word>>&` /
    `std::unique_ptr<UserInterface<word>>&`), matching the lattigo backend's
    `__configure` convention.

    After this pass, the module no longer depends on the CKKS dialect.
  }];

  let options = [
    Option<"entryFunction", "entry-function", "std::string", "", "Entry function to configure (auto-detected if empty)">,
    Option<"numCtsLevels", "num-cts-levels", "int", /*default=*/"3", "Bootstrap CoeffToSlot (CtS) level budget (cf. OpenFHE level-budget-encode); only used when the program bootstraps">,
    Option<"numStcLevels", "num-stc-levels", "int", /*default=*/"3", "Bootstrap SlotToCoeff (StC) level budget (cf. OpenFHE level-budget-decode); only used when the program bootstraps">,
    Option<"logMessageRatio", "log-message-ratio", "int", /*default=*/"-1", "Bootstrap message headroom passed to CHEDDAR's BootParameter (governs EvalMod precision). -1 uses CHEDDAR's default headroom of 5; only used when the program bootstraps">,
    Option<"prepareRotationKeysAtUseLevels", "prepare-rotation-keys-at-use-levels", "bool", /*default=*/"false", "Prepare each rotation key at the level of the operation that uses it instead of once at the maximum level">,
  ];

  let dependentDialects = [
    "mlir::heir::cheddar::CheddarDialect",
    "mlir::func::FuncDialect",
    "mlir::tensor::TensorDialect",
  ];
}

#endif  // LIB_DIALECT_CHEDDAR_TRANSFORMS_CONFIGURECRYPTOCONTEXT_TD_
