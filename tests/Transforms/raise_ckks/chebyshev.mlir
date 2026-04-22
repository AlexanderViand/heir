// RUN: heir-opt --raise-ckks %s | FileCheck %s

// Test raising orion.chebyshev back to polynomial.eval.

!Z1099502714881_i64 = !mod_arith.int<1099502714881 : i64>
!Z1099503370241_i64 = !mod_arith.int<1099503370241 : i64>
!Z36028797019488257_i64 = !mod_arith.int<36028797019488257 : i64>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 1099511627776>
#key = #lwe.key<>
#modulus_chain = #lwe.modulus_chain<elements = <36028797019488257 : i64, 1099503370241 : i64, 1099502714881 : i64>, current = 2>
#ring_f64 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**16>>
!rns = !rns.rns<!Z36028797019488257_i64, !Z1099503370241_i64, !Z1099502714881_i64>
#ring_rns = #polynomial.ring<coefficientType = !rns, polynomialModulus = <1 + x**16>>
#ciphertext_space = #lwe.ciphertext_space<ring = #ring_rns, encryption_type = mix>
!ct = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space, key = #key, modulus_chain = #modulus_chain>

module attributes {scheme.ckks, ckks.schemeParam = #ckks.scheme_param<logN = 4, Q = [36028797019488257, 1099503370241, 1099502714881], P = [36028797019488257], logDefaultScale = 40>} {
  func.func @chebyshev(%ct: !ct) -> !ct {
    %ct_0 = orion.chebyshev %ct {coefficients = [0.0, 0.75, 0.0, 0.25], domain_end = 1.0 : f64, domain_start = -1.0 : f64} : (!ct) -> !ct
    return %ct_0 : !ct
  }
}

// CHECK: module attributes
// CHECK-SAME: scheme.ckks
// CHECK-NOT: ckks.schemeParam
// CHECK: func.func @chebyshev(%[[ARG:.*]]: tensor<8xf64> {secret.secret})
// CHECK: polynomial.eval
// CHECK-SAME: typed_chebyshev_polynomial
// CHECK-SAME: domain_lower = -1.0
// CHECK-SAME: domain_upper = 1.0
// CHECK: return
