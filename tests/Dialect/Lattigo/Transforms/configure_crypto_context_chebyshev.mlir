// RUN: heir-opt --lattigo-configure-crypto-context=entry-function=cheb %s | FileCheck %s

// A chebyshev evaluation relinearizes internally (lattigo's
// polynomial.Evaluate squares the ciphertext while building its power basis),
// so a relinearization key must be generated even without an explicit
// relinearize op.

!eval = !lattigo.ckks.polynomial_evaluator
!ct = !lattigo.rlwe.ciphertext

module attributes {scheme.ckks} {
  func.func @cheb(%eval : !eval, %ct : !ct) -> !ct {
    %res = lattigo.ckks.chebyshev %eval, %ct {coefficients = [1.0, 0.5], targetScale = 1073741824, domain = array<f64: -2.0, 2.0>} : (!eval, !ct) -> !ct
    return %res : !ct
  }
}

// CHECK: @cheb
// CHECK: @cheb__configure
// CHECK: lattigo.rlwe.gen_relinearization_key
