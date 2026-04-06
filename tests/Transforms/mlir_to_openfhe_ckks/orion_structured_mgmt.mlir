// Test management of structured ops (polynomial.eval, diagonal_matvec) at arith level.
// These ops should be preserved through management with correct level_cost_ub annotations.

// Run through wrap-generic + annotate-orion + mgmt insertion (but NOT secret-to-ckks)
// to verify that structured ops are preserved with correct annotations.
// RUN: heir-opt --wrap-generic --annotate-orion --secret-insert-mgmt-ckks --split-input-file %s | FileCheck %s --check-prefix=LINEAR
// RUN: heir-opt --wrap-generic --annotate-orion --secret-insert-mgmt-ckks --split-input-file %s | FileCheck %s --check-prefix=CHEB

// LINEAR: secret.generic
// LINEAR: tensor_ext.diagonal_matvec
// LINEAR-SAME: orion.level_cost_ub = 0

func.func @linear_transform(%input: tensor<8xf64> {secret.secret}, %diags: tensor<2x8xf64>) -> tensor<8xf64> {
  %0 = tensor_ext.diagonal_matvec %input, %diags {diagonal_indices = array<i32: 0, 1>, slots = 8 : i64} : (tensor<8xf64>, tensor<2x8xf64>) -> tensor<8xf64>
  return %0 : tensor<8xf64>
}

// -----

// CHEB: secret.generic
// CHEB: polynomial.eval
// CHEB-SAME: orion.level_cost_ub = 2

#ring_f64 = #polynomial.ring<coefficientType = f64>
!poly = !polynomial.polynomial<ring = #ring_f64>

func.func @chebyshev(%input: tensor<8xf64> {secret.secret}) -> tensor<8xf64> {
  %0 = polynomial.eval #polynomial<typed_chebyshev_polynomial <[0.0, 0.75, 0.0, 0.25]> : !poly>, %input {domain_lower = -1.0 : f64, domain_upper = 1.0 : f64} : tensor<8xf64>
  return %0 : tensor<8xf64>
}
