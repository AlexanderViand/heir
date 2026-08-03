// RUN: heir-opt %s --lwe-to-cheddar | FileCheck %s

// Ct-pt plain ops. cheddar requires the ciphertext as the first operand, so the
// converter puts it first. Addition is commutative, so a swap is fine. But
// subtraction is NOT: cheddar.sub_plain computes `ct - pt`, so `pt - ct` must be
// lowered as `(-ct) + pt` (negate then add_plain) -- a blind operand swap would
// silently flip the sign.

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 1099511627776>
#key = #lwe.key<>
#modulus_chain_L10_C10 = #lwe.modulus_chain<elements = <36028797019488257 : i64, 1099512938497 : i64, 1099510054913 : i64, 1099507695617 : i64, 1099515691009 : i64, 1099516870657 : i64, 1099506515969 : i64, 1099504549889 : i64, 1099503894529 : i64, 1099503370241 : i64, 1099502714881 : i64>, current = 10>
#ring_f64_1_x65536 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**65536>>
!rns_L10 = !rns.rns<!mod_arith.int<36028797019488257 : i64>, !mod_arith.int<1099512938497 : i64>, !mod_arith.int<1099510054913 : i64>, !mod_arith.int<1099507695617 : i64>, !mod_arith.int<1099515691009 : i64>, !mod_arith.int<1099516870657 : i64>, !mod_arith.int<1099506515969 : i64>, !mod_arith.int<1099504549889 : i64>, !mod_arith.int<1099503894529 : i64>, !mod_arith.int<1099503370241 : i64>, !mod_arith.int<1099502714881 : i64>>
#ring_rns_L10_1_x65536 = #polynomial.ring<coefficientType = !rns_L10, polynomialModulus = <1 + x**65536>>
#ciphertext_space_L10 = #lwe.ciphertext_space<ring = #ring_rns_L10_1_x65536, encryption_type = mix>
!ct_L10 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x65536, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L10, key = #key, modulus_chain = #modulus_chain_L10_C10>
!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_f64_1_x65536, encoding = #inverse_canonical_encoding>>

module attributes {scheme.ckks, ckks.schemeParam = #ckks.scheme_param<logN = 16, Q = [36028797019488257, 1099512938497, 1099510054913, 1099507695617, 1099515691009, 1099516870657, 1099506515969, 1099504549889, 1099503894529, 1099503370241, 1099502714881], P = [2305843009211596801, 2305843009210023937, 2305843009208713217], logDefaultScale = 40>} {
  // ct - pt: direct sub_plain, no negation.
  // CHECK: func @ct_minus_pt
  func.func @ct_minus_pt(%ct: !ct_L10, %pt: !pt) -> !ct_L10 {
    // CHECK: cheddar.sub_plain
    // CHECK-NOT: cheddar.neg
    %0 = lwe.rsub_plain %ct, %pt : (!ct_L10, !pt) -> !ct_L10
    return %0 : !ct_L10
  }

  // pt - ct: must be (-ct) + pt, NOT a sign-flipped sub_plain.
  // CHECK: func @pt_minus_ct
  func.func @pt_minus_ct(%ct: !ct_L10, %pt: !pt) -> !ct_L10 {
    // CHECK-NOT: cheddar.sub_plain
    // CHECK: cheddar.neg
    // CHECK: cheddar.add_plain
    %0 = lwe.rsub_plain %pt, %ct : (!pt, !ct_L10) -> !ct_L10
    return %0 : !ct_L10
  }

  // Addition is commutative: pt + ct lowers to a single add_plain.
  // CHECK: func @pt_plus_ct
  func.func @pt_plus_ct(%ct: !ct_L10, %pt: !pt) -> !ct_L10 {
    // CHECK: cheddar.add_plain
    %0 = lwe.radd_plain %pt, %ct : (!pt, !ct_L10) -> !ct_L10
    return %0 : !ct_L10
  }
}
