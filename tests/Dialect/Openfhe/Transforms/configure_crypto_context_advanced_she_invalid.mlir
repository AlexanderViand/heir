// RUN: not heir-opt --openfhe-configure-crypto-context="entry-function=invalid_chebyshev scaling-technique=fixed-manual" %s 2>&1 | FileCheck %s

!cc = !openfhe.crypto_context
!ct = !openfhe.ciphertext

module attributes {backend.openfhe, scheme.ckks} {
  func.func @invalid_chebyshev(%arg0: !cc, %arg1: !ct) -> !ct {
    %0 = openfhe.chebyshev_series %arg0, %arg1 {coefficients = [0.0, 1.0], domainStart = -1.000000e+00 : f64, domainEnd = 1.000000e+00 : f64} : (!cc, !ct) -> !ct
    return %0 : !ct
  }
}

// CHECK: error: 'func.func' op CKKS OpenFHE configuration for bootstrap, linear_transform, and chebyshev_series requires a precomputed `ckks.schemeParam` or an explicit `--openfhe-configure-crypto-context=mul-depth=` override
