// RUN: heir-opt --openfhe-configure-crypto-context="entry-function=needs_advanced_she scaling-technique=fixed-manual" --split-input-file %s | FileCheck %s --check-prefix=FIXED-MANUAL-ADV --check-prefix=FIXED-MANUAL-RESCALE
// RUN: heir-opt --openfhe-configure-crypto-context="entry-function=needs_advanced_she scaling-technique=flexible-auto" --split-input-file %s | FileCheck %s --check-prefix=FLEXIBLE-AUTO-ADV --check-prefix=FLEXIBLE-AUTO-RESCALE
// RUN: heir-opt --openfhe-configure-crypto-context="entry-function=needs_advanced_she scaling-technique=flexible-auto-ext" --split-input-file %s | FileCheck %s --check-prefix=FLEXIBLE-AUTO-EXT-ADV --check-prefix=FLEXIBLE-AUTO-EXT-RESCALE

!cc = !openfhe.crypto_context
!ct = !openfhe.ciphertext

module attributes {
  backend.openfhe,
  scheme.ckks,
  ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [536903681, 67043329], P = [536952833, 536690689], logDefaultScale = 26>
} {
  func.func @needs_advanced_she(%arg0: !cc, %arg1: !ct) -> !ct {
    %0 = openfhe.chebyshev_series %arg0, %arg1 {coefficients = [0.0, 0.75, 0.0, 0.25], domainStart = -1.000000e+00 : f64, domainEnd = 1.000000e+00 : f64} : (!cc, !ct) -> !ct
    return %0 : !ct
  }
}

// FIXED-MANUAL-ADV: func.func @needs_advanced_she__generate_crypto_context
// FIXED-MANUAL-ADV: openfhe.gen_params
// FIXED-MANUAL-ADV-SAME: scalingTechnique = "fixed-manual"
// FIXED-MANUAL-ADV: %{{.*}} = openfhe.gen_context %{{.*}} {supportFHE = true}
// FLEXIBLE-AUTO-ADV: func.func @needs_advanced_she__generate_crypto_context
// FLEXIBLE-AUTO-ADV: openfhe.gen_params
// FLEXIBLE-AUTO-ADV-SAME: scalingTechnique = "flexible-auto"
// FLEXIBLE-AUTO-ADV: %{{.*}} = openfhe.gen_context %{{.*}} {supportFHE = true}
// FLEXIBLE-AUTO-EXT-ADV: func.func @needs_advanced_she__generate_crypto_context
// FLEXIBLE-AUTO-EXT-ADV: openfhe.gen_params
// FLEXIBLE-AUTO-EXT-ADV-SAME: scalingTechnique = "flexible-auto-ext"
// FLEXIBLE-AUTO-EXT-ADV: %{{.*}} = openfhe.gen_context %{{.*}} {supportFHE = true}

// -----

!cc = !openfhe.crypto_context
!ct = !openfhe.ciphertext

module attributes {
  backend.openfhe,
  scheme.ckks,
  ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [536903681, 67043329, 66994177], P = [536952833, 536690689], logDefaultScale = 26>
} {
  func.func @needs_domain_rescaling(%arg0: !cc, %arg1: !ct) -> !ct {
    %0 = openfhe.chebyshev_series %arg0, %arg1 {coefficients = [0.0, 0.75, 0.0, 0.25], domainStart = 0.000000e+00 : f64, domainEnd = 2.000000e+00 : f64} : (!cc, !ct) -> !ct
    return %0 : !ct
  }
}

// FIXED-MANUAL-RESCALE: func.func @needs_domain_rescaling__generate_crypto_context
// FIXED-MANUAL-RESCALE: openfhe.gen_params
// FIXED-MANUAL-RESCALE-SAME: scalingTechnique = "fixed-manual"
// FIXED-MANUAL-RESCALE: %{{.*}} = openfhe.gen_context %{{.*}} {supportFHE = true}
// FLEXIBLE-AUTO-RESCALE: func.func @needs_domain_rescaling__generate_crypto_context
// FLEXIBLE-AUTO-RESCALE: openfhe.gen_params
// FLEXIBLE-AUTO-RESCALE-SAME: scalingTechnique = "flexible-auto"
// FLEXIBLE-AUTO-RESCALE: %{{.*}} = openfhe.gen_context %{{.*}} {supportFHE = true}
// FLEXIBLE-AUTO-EXT-RESCALE: func.func @needs_domain_rescaling__generate_crypto_context
// FLEXIBLE-AUTO-EXT-RESCALE: openfhe.gen_params
// FLEXIBLE-AUTO-EXT-RESCALE-SAME: scalingTechnique = "flexible-auto-ext"
// FLEXIBLE-AUTO-EXT-RESCALE: %{{.*}} = openfhe.gen_context %{{.*}} {supportFHE = true}
