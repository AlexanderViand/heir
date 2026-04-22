// RUN: heir-opt --openfhe-configure-crypto-context="entry-function=nested_linear_transform scaling-technique=fixed-manual" --split-input-file %s | FileCheck %s --check-prefix=LINEAR
// RUN: heir-opt --openfhe-configure-crypto-context="entry-function=nested_chebyshev scaling-technique=fixed-manual" --split-input-file %s | FileCheck %s --check-prefix=CHEB

!cc = !openfhe.crypto_context
!ct = !openfhe.ciphertext

module attributes {
  backend.openfhe,
  scheme.ckks,
  ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [536903681, 67043329, 66994177], P = [536952833, 536690689], logDefaultScale = 26>
} {
  func.func @helper_linear_transform(%arg0: !cc, %arg1: !ct, %arg2: tensor<1x8xf64>) -> !ct {
    %0 = openfhe.linear_transform %arg0, %arg1, %arg2 {diagonal_indices = array<i32: 0>, plaintextLevel = 2 : i64, logBabyStepGiantStepRatio = 0 : i64} : (!cc, !ct, tensor<1x8xf64>) -> !ct
    return %0 : !ct
  }

  func.func @nested_linear_transform(%arg0: !cc, %arg1: !ct, %arg2: tensor<1x8xf64>) -> !ct {
    %0 = call @helper_linear_transform(%arg0, %arg1, %arg2) : (!cc, !ct, tensor<1x8xf64>) -> !ct
    return %0 : !ct
  }
}

// LINEAR: func.func @nested_linear_transform__generate_crypto_context
// LINEAR: openfhe.gen_params
// LINEAR-SAME: mulDepth = 3

// -----

!cc = !openfhe.crypto_context
!ct = !openfhe.ciphertext

module attributes {
  backend.openfhe,
  scheme.ckks,
  ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [536903681, 67043329], P = [536952833, 536690689], logDefaultScale = 26>
} {
  func.func @helper_chebyshev(%arg0: !cc, %arg1: !ct) -> !ct {
    %0 = openfhe.chebyshev_series %arg0, %arg1 {coefficients = [0.0, 0.75, 0.0, 0.25], domainStart = -1.000000e+00 : f64, domainEnd = 1.000000e+00 : f64} : (!cc, !ct) -> !ct
    return %0 : !ct
  }

  func.func @nested_chebyshev(%arg0: !cc, %arg1: !ct) -> !ct {
    %0 = call @helper_chebyshev(%arg0, %arg1) : (!cc, !ct) -> !ct
    return %0 : !ct
  }
}

// CHEB: func.func @nested_chebyshev__generate_crypto_context
// CHEB: openfhe.gen_params
// CHEB-SAME: mulDepth = 2
