// RUN: heir-opt --lwe-to-cheddar %s | FileCheck %s

#enc = #lwe.inverse_canonical_encoding<scaling_factor = 45>
#key = #lwe.key<>
#chain_in = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64, 35184372088833 : i64>, current = 2>
#chain_out = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64, 35184372088833 : i64>, current = 0>
#rf = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_in = !rns.rns<!mod_arith.int<36028797018652673 : i64>, !mod_arith.int<35184372121601 : i64>, !mod_arith.int<35184372088833 : i64>>
!rns_out = !rns.rns<!mod_arith.int<36028797018652673 : i64>>
#rr_in = #polynomial.ring<coefficientType = !rns_in, polynomialModulus = <1 + x**1024>>
#rr_out = #polynomial.ring<coefficientType = !rns_out, polynomialModulus = <1 + x**1024>>
#cs_in = #lwe.ciphertext_space<ring = #rr_in, encryption_type = mix>
#cs_out = #lwe.ciphertext_space<ring = #rr_out, encryption_type = mix>
!ct_in = !lwe.lwe_ciphertext<plaintext_space = <ring = #rf, encoding = #enc>, ciphertext_space = #cs_in, key = #key, modulus_chain = #chain_in>
!ct_out = !lwe.lwe_ciphertext<plaintext_space = <ring = #rf, encoding = #enc>, ciphertext_space = #cs_out, key = #key, modulus_chain = #chain_out>

module attributes {backend.cheddar, ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797018652673, 35184372121601, 35184372088833], P = [1152921504606994433], logDefaultScale = 45, encryptionTechnique = extended>, scheme.ckks} {
  // CHECK: func.func @eval_chebyshev
  // CHECK: cheddar.eval_poly
  // CHECK-SAME: coefficients = [0.000000e+00, 7.500000e-01, 0.000000e+00, 2.500000e-01]
  // CHECK-SAME: levelConsumption = 2 : i64
  func.func @eval_chebyshev(%ct: !ct_in) -> !ct_out {
    %0 = kernel.eval_chebyshev %ct {coefficients = [0.0 : f64, 0.75 : f64, 0.0 : f64, 0.25 : f64]} : !ct_in -> !ct_out
    return %0 : !ct_out
  }
}
