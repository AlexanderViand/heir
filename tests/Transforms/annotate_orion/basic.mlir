// RUN: heir-opt --split-input-file --annotate-orion %s | FileCheck %s

// -----

!Z536903681_i64 = !mod_arith.int<536903681 : i64>
!Z66813953_i64 = !mod_arith.int<66813953 : i64>
!Z66961409_i64 = !mod_arith.int<66961409 : i64>
!Z66994177_i64 = !mod_arith.int<66994177 : i64>
!Z67043329_i64 = !mod_arith.int<67043329 : i64>
!Z67239937_i64 = !mod_arith.int<67239937 : i64>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 67108864>
#key = #lwe.key<>
#modulus_chain_L5_C5 = #lwe.modulus_chain<elements = <536903681 : i64, 67043329 : i64, 66994177 : i64, 67239937 : i64, 66961409 : i64, 66813953 : i64>, current = 5>
#ring_f64_1_x8192 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**8192>>
!rns_L5 = !rns.rns<!Z536903681_i64, !Z67043329_i64, !Z66994177_i64, !Z67239937_i64, !Z66961409_i64, !Z66813953_i64>
#ring_rns_L5_1_x8192 = #polynomial.ring<coefficientType = !rns_L5, polynomialModulus = <1 + x**8192>>
#ciphertext_space_L5 = #lwe.ciphertext_space<ring = #ring_rns_L5_1_x8192, encryption_type = mix>
!ct_L5 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x8192, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L5, key = #key, modulus_chain = #modulus_chain_L5_C5>

// CHECK: orion.linear_transform
// CHECK-SAME: orion.impl_style = "diagonal-bsgs"
// CHECK-SAME: orion.level_cost_ub = 0 : i64
module {
  func.func @linear_transform(%ct: !ct_L5, %arg0: tensor<2x4096xf64>) -> !ct_L5 {
    %ct_0 = orion.linear_transform %ct, %arg0 {block_col = 0 : i32, block_row = 0 : i32, bsgs_ratio = 2.000000e+00 : f64, diagonal_count = 2 : i32, orion_level = 5 : i32, slots = 4096 : i32, diagonal_indices = array<i32: 0, 1>} : (!ct_L5, tensor<2x4096xf64>) -> !ct_L5
    return %ct_0 : !ct_L5
  }
}

// -----

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
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 1099511627776>
#key = #lwe.key<>
#modulus_chain_L10_C10 = #lwe.modulus_chain<elements = <36028797019488257 : i64, 1099512938497 : i64, 1099510054913 : i64, 1099507695617 : i64, 1099515691009 : i64, 1099516870657 : i64, 1099506515969 : i64, 1099504549889 : i64, 1099503894529 : i64, 1099503370241 : i64, 1099502714881 : i64>, current = 10>
#ring_f64_1_x65536 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**65536>>
!rns_L10 = !rns.rns<!Z36028797019488257_i64, !Z1099512938497_i64, !Z1099510054913_i64, !Z1099507695617_i64, !Z1099515691009_i64, !Z1099516870657_i64, !Z1099506515969_i64, !Z1099504549889_i64, !Z1099503894529_i64, !Z1099503370241_i64, !Z1099502714881_i64>
#ring_rns_L10_1_x65536 = #polynomial.ring<coefficientType = !rns_L10, polynomialModulus = <1 + x**65536>>
#ciphertext_space_L10 = #lwe.ciphertext_space<ring = #ring_rns_L10_1_x65536, encryption_type = mix>
!ct_L10 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x65536, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L10, key = #key, modulus_chain = #modulus_chain_L10_C10>

// CHECK: orion.chebyshev
// CHECK-SAME: orion.impl_style = "bsgs"
// CHECK-SAME: orion.level_cost_ub = 2 : i64
module {
  func.func @chebyshev(%ct: !ct_L10) -> !ct_L10 {
    %ct_0 = orion.chebyshev %ct {coefficients = [0.0, 0.75, 0.0, 0.25], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L10) -> !ct_L10
    return %ct_0 : !ct_L10
  }
}
