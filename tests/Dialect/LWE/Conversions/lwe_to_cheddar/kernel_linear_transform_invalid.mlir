// RUN: not heir-opt --lwe-to-cheddar %s 2>&1 | FileCheck %s

#enc = #lwe.inverse_canonical_encoding<scaling_factor = 45>
#key = #lwe.key<>
#chain = #lwe.modulus_chain<elements = <36028797018652673 : i64>, current = 0>
#rf = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns = !rns.rns<!mod_arith.int<36028797018652673 : i64>>
#rr = #polynomial.ring<coefficientType = !rns, polynomialModulus = <1 + x**1024>>
#cs = #lwe.ciphertext_space<ring = #rr, encryption_type = mix>
!ct = !lwe.lwe_ciphertext<plaintext_space = <ring = #rf, encoding = #enc>, ciphertext_space = #cs, key = #key, modulus_chain = #chain>

module attributes {backend.cheddar, ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797018652673], P = [1152921504606994433], logDefaultScale = 45, encryptionTechnique = extended>, scheme.ckks} {
  // CHECK: error: 'kernel.linear_transform' op scale-snu CHEDDAR linear transforms require an input level above zero
  func.func @level_zero(%ct: !ct) -> !ct {
    %diagonals = arith.constant dense<1.0> : tensor<2x8xf64>
    %0 = kernel.linear_transform %ct, %diagonals {diagonal_indices = array<i64: 0, 1>} : !ct, tensor<2x8xf64> -> !ct
    return %0 : !ct
  }
}
