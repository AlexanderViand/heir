// RUN: heir-opt --openfhe-configure-crypto-context="entry-function=needs_advanced_she scaling-technique=fixed-manual" --split-input-file %s | FileCheck %s --check-prefix=FIXED-MANUAL
// RUN: heir-opt --openfhe-configure-crypto-context="entry-function=needs_advanced_she scaling-technique=fixed-auto" --split-input-file %s | FileCheck %s --check-prefix=FIXED-AUTO
// RUN: heir-opt --openfhe-configure-crypto-context="entry-function=needs_advanced_she scaling-technique=flexible-auto" --split-input-file %s | FileCheck %s --check-prefix=FLEXIBLE-AUTO
// RUN: heir-opt --openfhe-configure-crypto-context="entry-function=needs_advanced_she scaling-technique=flexible-auto-ext" --split-input-file %s | FileCheck %s --check-prefix=FLEXIBLE-AUTO-EXT

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

// FIXED-MANUAL: func.func @needs_advanced_she__generate_crypto_context
// FIXED-MANUAL: openfhe.gen_params
// FIXED-MANUAL-SAME: mulDepth = 2
// FIXED-MANUAL: %{{.*}} = openfhe.gen_context %{{.*}} {supportFHE = true}
// FIXED-AUTO: func.func @needs_advanced_she__generate_crypto_context
// FIXED-AUTO: openfhe.gen_params
// FIXED-AUTO-SAME: mulDepth = 2
// FIXED-AUTO: %{{.*}} = openfhe.gen_context %{{.*}} {supportFHE = true}
// FLEXIBLE-AUTO: func.func @needs_advanced_she__generate_crypto_context
// FLEXIBLE-AUTO: openfhe.gen_params
// FLEXIBLE-AUTO-SAME: mulDepth = 2
// FLEXIBLE-AUTO: %{{.*}} = openfhe.gen_context %{{.*}} {supportFHE = true}
// FLEXIBLE-AUTO-EXT: func.func @needs_advanced_she__generate_crypto_context
// FLEXIBLE-AUTO-EXT: openfhe.gen_params
// FLEXIBLE-AUTO-EXT-SAME: mulDepth = 2
// FLEXIBLE-AUTO-EXT: %{{.*}} = openfhe.gen_context %{{.*}} {supportFHE = true}

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

// FIXED-MANUAL: func.func @needs_domain_rescaling__generate_crypto_context
// FIXED-MANUAL: openfhe.gen_params
// FIXED-MANUAL-SAME: mulDepth = 3
// FIXED-MANUAL: %{{.*}} = openfhe.gen_context %{{.*}} {supportFHE = true}
// FIXED-AUTO: func.func @needs_domain_rescaling__generate_crypto_context
// FIXED-AUTO: openfhe.gen_params
// FIXED-AUTO-SAME: mulDepth = 3
// FIXED-AUTO: %{{.*}} = openfhe.gen_context %{{.*}} {supportFHE = true}
// FLEXIBLE-AUTO: func.func @needs_domain_rescaling__generate_crypto_context
// FLEXIBLE-AUTO: openfhe.gen_params
// FLEXIBLE-AUTO-SAME: mulDepth = 3
// FLEXIBLE-AUTO: %{{.*}} = openfhe.gen_context %{{.*}} {supportFHE = true}
// FLEXIBLE-AUTO-EXT: func.func @needs_domain_rescaling__generate_crypto_context
// FLEXIBLE-AUTO-EXT: openfhe.gen_params
// FLEXIBLE-AUTO-EXT-SAME: mulDepth = 3
// FLEXIBLE-AUTO-EXT: %{{.*}} = openfhe.gen_context %{{.*}} {supportFHE = true}
