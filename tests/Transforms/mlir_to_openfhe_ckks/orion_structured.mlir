// Test lowering Orion CKKS-level IR to OpenFHE via the raise path.

// RUN: heir-opt --raise-ckks '--mlir-to-ckks=ciphertext-degree=4096 enable-arithmetization=false preserve-structured-ops=true scaling-mod-bits=26 first-mod-bits=30 openfhe-scaling-technique=fixed-manual' '--scheme-to-openfhe=openfhe-scaling-technique=fixed-manual' --split-input-file %s | FileCheck %s --check-prefix=LINEAR
// RUN: heir-opt --raise-ckks '--mlir-to-ckks=ciphertext-degree=32768 enable-arithmetization=false preserve-structured-ops=true scaling-mod-bits=40 openfhe-scaling-technique=fixed-manual' '--scheme-to-openfhe=openfhe-scaling-technique=fixed-manual' --split-input-file %s | FileCheck %s --check-prefix=CHEB

// LINEAR: @linear_transform
// LINEAR-NOT: orion.linear_transform
// LINEAR: openfhe.linear_transform

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
module attributes {scheme.ckks, ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [536903681, 67043329, 66994177, 67239937, 66961409, 66813953], P = [536952833, 536690689], logDefaultScale = 26>} {
  func.func @linear_transform(%ct: !ct_L5, %arg0: tensor<2x4096xf64>) -> !ct_L5 {
    %ct_0 = orion.linear_transform %ct, %arg0 {block_col = 0 : i32, block_row = 0 : i32, bsgs_ratio = 2.000000e+00 : f64, diagonal_count = 2 : i32, orion_level = 5 : i32, slots = 4096 : i32, diagonal_indices = array<i32: 0, 1>} : (!ct_L5, tensor<2x4096xf64>) -> !ct_L5
    return %ct_0 : !ct_L5
  }
}

// -----

// CHEB: @chebyshev
// CHEB-NOT: orion.chebyshev
// CHEB: openfhe.chebyshev_series

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
#inverse_canonical_encoding_2 = #lwe.inverse_canonical_encoding<scaling_factor = 1099511627776>
#key_2 = #lwe.key<>
#modulus_chain_L10_C10 = #lwe.modulus_chain<elements = <36028797019488257 : i64, 1099512938497 : i64, 1099510054913 : i64, 1099507695617 : i64, 1099515691009 : i64, 1099516870657 : i64, 1099506515969 : i64, 1099504549889 : i64, 1099503894529 : i64, 1099503370241 : i64, 1099502714881 : i64>, current = 10>
#ring_f64_1_x65536 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**65536>>
!rns_L10 = !rns.rns<!Z36028797019488257_i64, !Z1099512938497_i64, !Z1099510054913_i64, !Z1099507695617_i64, !Z1099515691009_i64, !Z1099516870657_i64, !Z1099506515969_i64, !Z1099504549889_i64, !Z1099503894529_i64, !Z1099503370241_i64, !Z1099502714881_i64>
#ring_rns_L10_1_x65536 = #polynomial.ring<coefficientType = !rns_L10, polynomialModulus = <1 + x**65536>>
#ciphertext_space_L10 = #lwe.ciphertext_space<ring = #ring_rns_L10_1_x65536, encryption_type = mix>
!ct_L10 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x65536, encoding = #inverse_canonical_encoding_2>, ciphertext_space = #ciphertext_space_L10, key = #key_2, modulus_chain = #modulus_chain_L10_C10>
module attributes {scheme.ckks, ckks.schemeParam = #ckks.scheme_param<logN = 16, Q = [36028797019488257, 1099512938497, 1099510054913, 1099507695617, 1099515691009, 1099516870657, 1099506515969, 1099504549889, 1099503894529, 1099503370241, 1099502714881], P = [2305843009211596801, 2305843009210023937, 2305843009208713217], logDefaultScale = 40>} {
  func.func @chebyshev(%ct: !ct_L10) -> !ct_L10 {
    %ct_0 = orion.chebyshev %ct {coefficients = [0.0, 0.75, 0.0, 0.25], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L10) -> !ct_L10
    return %ct_0 : !ct_L10
  }
}
