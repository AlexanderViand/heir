// RUN: heir-opt "--openfhe-configure-crypto-context=entry-function=override scaling-technique=flexible-auto-ext" --split-input-file %s | FileCheck %s --check-prefix=OVERRIDE
// RUN: heir-opt --openfhe-configure-crypto-context=entry-function=inferred --split-input-file %s | FileCheck %s --check-prefix=INFERRED

!ct = !openfhe.ciphertext

module attributes {backend.openfhe, scheme.ckks} {
  func.func @override(%arg0: !openfhe.crypto_context, %arg1: !ct) -> !ct {
    %0 = openfhe.mod_reduce %arg0, %arg1 : (!openfhe.crypto_context, !ct) -> !ct
    return %0 : !ct
  }
}

// OVERRIDE: func.func @override__generate_crypto_context
// OVERRIDE: openfhe.gen_params
// OVERRIDE-SAME: scalingTechnique = "flexible-auto-ext"

// -----

!ct = !openfhe.ciphertext

module attributes {
  backend.openfhe,
  scheme.ckks,
  ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [536903681, 67043329, 66994177, 67239937, 66961409, 66813953], P = [536952833, 536690689], logDefaultScale = 26>,
  openfhe.scaling_technique = "fixed-manual"
} {
  func.func @inferred(%arg0: !openfhe.crypto_context, %arg1: !ct) -> !ct {
    %0 = openfhe.mod_reduce %arg0, %arg1 : (!openfhe.crypto_context, !ct) -> !ct
    return %0 : !ct
  }
}

// INFERRED: func.func @inferred__generate_crypto_context
// INFERRED: openfhe.gen_params
// INFERRED-SAME: batchSize = 4096
// INFERRED-SAME: firstModSize = 30
// INFERRED-SAME: ringDim = 8192
// INFERRED-SAME: scalingModSize = 26
// INFERRED-SAME: scalingTechnique = "fixed-manual"
