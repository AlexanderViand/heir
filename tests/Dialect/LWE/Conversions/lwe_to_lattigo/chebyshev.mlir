// RUN: heir-opt --annotate-orion --lwe-to-lattigo --split-input-file %s | FileCheck %s

// CHECK: func.func @chebyshev_nominal
// CHECK: lattigo.ckks.chebyshev
// CHECK-SAME: targetScale = 1099511627776 : i64

!Z1099502714881_i64 = !mod_arith.int<1099502714881 : i64>
!Z1099503370241_i64 = !mod_arith.int<1099503370241 : i64>
!Z1099503894529_i64 = !mod_arith.int<1099503894529 : i64>
!Z1099504549889_i64 = !mod_arith.int<1099504549889 : i64>
!Z1099506515969_i64 = !mod_arith.int<1099506515969 : i64>
!Z1099507695617_i64 = !mod_arith.int<1099507695617 : i64>
!Z1099510054913_i64 = !mod_arith.int<1099510054913 : i64>
!Z1099512938497_i64 = !mod_arith.int<1099512938497 : i64>
!Z1099515691009_i64 = !mod_arith.int<1099515691009 : i64>
!Z1099516870657_i64 = !mod_arith.int<1099516870657 : i64>
!Z36028797019488257_i64 = !mod_arith.int<36028797019488257 : i64>
#inverse_canonical_encoding_nominal = #lwe.inverse_canonical_encoding<scaling_factor = 34359738368>
#key = #lwe.key<>
#modulus_chain_L10_C10 = #lwe.modulus_chain<elements = <36028797019488257 : i64, 1099512938497 : i64, 1099510054913 : i64, 1099507695617 : i64, 1099515691009 : i64, 1099516870657 : i64, 1099506515969 : i64, 1099504549889 : i64, 1099503894529 : i64, 1099503370241 : i64, 1099502714881 : i64>, current = 10>
#ring_f64_1_x65536 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**65536>>
!rns_L10 = !rns.rns<!Z36028797019488257_i64, !Z1099512938497_i64, !Z1099510054913_i64, !Z1099507695617_i64, !Z1099515691009_i64, !Z1099516870657_i64, !Z1099506515969_i64, !Z1099504549889_i64, !Z1099503894529_i64, !Z1099503370241_i64, !Z1099502714881_i64>
#ring_rns_L10_1_x65536 = #polynomial.ring<coefficientType = !rns_L10, polynomialModulus = <1 + x**65536>>
#ciphertext_space_L10 = #lwe.ciphertext_space<ring = #ring_rns_L10_1_x65536, encryption_type = mix>
!ct_L10_nominal = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x65536, encoding = #inverse_canonical_encoding_nominal>, ciphertext_space = #ciphertext_space_L10, key = #key, modulus_chain = #modulus_chain_L10_C10>
module attributes {scheme.ckks, ckks.schemeParam = #ckks.scheme_param<logN = 16, Q = [36028797019488257, 1099512938497, 1099510054913, 1099507695617, 1099515691009, 1099516870657, 1099506515969, 1099504549889, 1099503894529, 1099503370241, 1099502714881], P = [2305843009211596801, 2305843009210023937, 2305843009208713217], logDefaultScale = 40>} {
  func.func @chebyshev_nominal(%ct: !ct_L10_nominal) -> !ct_L10_nominal {
    %ct_0 = orion.chebyshev %ct {coefficients = [0.0, 0.75, 0.0, 0.25], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L10_nominal) -> !ct_L10_nominal
    return %ct_0 : !ct_L10_nominal
  }
}

// -----

// CHECK: func.func @chebyshev_precise
// CHECK: lattigo.ckks.chebyshev
// CHECK-SAME: targetScale = 34359738369 : i64

!Z1099502714881_i64 = !mod_arith.int<1099502714881 : i64>
!Z1099503370241_i64 = !mod_arith.int<1099503370241 : i64>
!Z1099503894529_i64 = !mod_arith.int<1099503894529 : i64>
!Z1099504549889_i64 = !mod_arith.int<1099504549889 : i64>
!Z1099506515969_i64 = !mod_arith.int<1099506515969 : i64>
!Z1099507695617_i64 = !mod_arith.int<1099507695617 : i64>
!Z1099510054913_i64 = !mod_arith.int<1099510054913 : i64>
!Z1099512938497_i64 = !mod_arith.int<1099512938497 : i64>
!Z1099515691009_i64 = !mod_arith.int<1099515691009 : i64>
!Z1099516870657_i64 = !mod_arith.int<1099516870657 : i64>
!Z36028797019488257_i64 = !mod_arith.int<36028797019488257 : i64>
#inverse_canonical_encoding_precise = #lwe.inverse_canonical_encoding<scaling_factor = 34359738369>
#key = #lwe.key<>
#modulus_chain_L10_C10 = #lwe.modulus_chain<elements = <36028797019488257 : i64, 1099512938497 : i64, 1099510054913 : i64, 1099507695617 : i64, 1099515691009 : i64, 1099516870657 : i64, 1099506515969 : i64, 1099504549889 : i64, 1099503894529 : i64, 1099503370241 : i64, 1099502714881 : i64>, current = 10>
#ring_f64_1_x65536 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**65536>>
!rns_L10 = !rns.rns<!Z36028797019488257_i64, !Z1099512938497_i64, !Z1099510054913_i64, !Z1099507695617_i64, !Z1099515691009_i64, !Z1099516870657_i64, !Z1099506515969_i64, !Z1099504549889_i64, !Z1099503894529_i64, !Z1099503370241_i64, !Z1099502714881_i64>
#ring_rns_L10_1_x65536 = #polynomial.ring<coefficientType = !rns_L10, polynomialModulus = <1 + x**65536>>
#ciphertext_space_L10 = #lwe.ciphertext_space<ring = #ring_rns_L10_1_x65536, encryption_type = mix>
!ct_L10_precise = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x65536, encoding = #inverse_canonical_encoding_precise>, ciphertext_space = #ciphertext_space_L10, key = #key, modulus_chain = #modulus_chain_L10_C10>
module attributes {ckks.scale_policy = "precise", scheme.ckks, ckks.schemeParam = #ckks.scheme_param<logN = 16, Q = [36028797019488257, 1099512938497, 1099510054913, 1099507695617, 1099515691009, 1099516870657, 1099506515969, 1099504549889, 1099503894529, 1099503370241, 1099502714881], P = [2305843009211596801, 2305843009210023937, 2305843009208713217], logDefaultScale = 40>} {
  func.func @chebyshev_precise(%ct: !ct_L10_precise) -> !ct_L10_precise {
    %ct_0 = orion.chebyshev %ct {coefficients = [0.0, 0.75, 0.0, 0.25], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L10_precise) -> !ct_L10_precise
    return %ct_0 : !ct_L10_precise
  }
}
