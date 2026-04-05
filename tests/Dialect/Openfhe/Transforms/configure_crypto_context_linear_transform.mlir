// RUN: heir-opt --openfhe-configure-crypto-context="entry-function=needs_lowered_aux scaling-technique=fixed-manual" %s | FileCheck %s --check-prefix=FIXED-MANUAL
// RUN: heir-opt --openfhe-configure-crypto-context="entry-function=needs_lowered_aux scaling-technique=flexible-auto" %s | FileCheck %s --check-prefix=FLEXIBLE-AUTO
// RUN: heir-opt --openfhe-configure-crypto-context="entry-function=needs_lowered_aux scaling-technique=flexible-auto-ext" %s | FileCheck %s --check-prefix=FLEXIBLE-AUTO-EXT

!cc = !openfhe.crypto_context
!ct = !openfhe.ciphertext

module attributes {
  backend.openfhe,
  scheme.ckks,
  ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [536903681, 67043329, 66994177], P = [536952833, 536690689], logDefaultScale = 26>
} {
  func.func @needs_lowered_aux(%arg0: !cc, %arg1: !ct, %arg2: tensor<1x8xf64>) -> !ct {
    %0 = openfhe.linear_transform %arg0, %arg1, %arg2 {diagonal_indices = array<i32: 0>, plaintextLevel = 2 : i64, logBabyStepGiantStepRatio = 0 : i64} : (!cc, !ct, tensor<1x8xf64>) -> !ct
    return %0 : !ct
  }
}

// FIXED-MANUAL: func.func @needs_lowered_aux__generate_crypto_context
// FIXED-MANUAL: openfhe.gen_params
// FIXED-MANUAL-SAME: scalingTechnique = "fixed-manual"
// FIXED-MANUAL: %{{.*}} = openfhe.gen_context %{{.*}} {supportFHE = true}
// FLEXIBLE-AUTO: func.func @needs_lowered_aux__generate_crypto_context
// FLEXIBLE-AUTO: openfhe.gen_params
// FLEXIBLE-AUTO-SAME: scalingTechnique = "flexible-auto"
// FLEXIBLE-AUTO: %{{.*}} = openfhe.gen_context %{{.*}} {supportFHE = true}
// FLEXIBLE-AUTO-EXT: func.func @needs_lowered_aux__generate_crypto_context
// FLEXIBLE-AUTO-EXT: openfhe.gen_params
// FLEXIBLE-AUTO-EXT-SAME: scalingTechnique = "flexible-auto-ext"
// FLEXIBLE-AUTO-EXT: %{{.*}} = openfhe.gen_context %{{.*}} {supportFHE = true}
