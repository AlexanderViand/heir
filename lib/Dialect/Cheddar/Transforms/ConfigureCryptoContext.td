#ifndef LIB_DIALECT_CHEDDAR_TRANSFORMS_CONFIGURECRYPTOCONTEXT_TD_
#define LIB_DIALECT_CHEDDAR_TRANSFORMS_CONFIGURECRYPTOCONTEXT_TD_

include "mlir/Pass/PassBase.td"

def CheddarConfigureCryptoContext : Pass<"cheddar-configure-crypto-context"> {
  let summary = "Configure CHEDDAR crypto context from scheme parameters.";

  let description = [{
    This pass reads the `ckks.schemeParam` module attribute, generates a
    `__configure` setup function that creates the CHEDDAR context, encoder,
    and user interface, and removes the CKKS module attributes.

    After this pass, the module no longer depends on the CKKS dialect.
  }];

  let dependentDialects = [
    "mlir::heir::cheddar::CheddarDialect",
  ];
}

#endif  // LIB_DIALECT_CHEDDAR_TRANSFORMS_CONFIGURECRYPTOCONTEXT_TD_
