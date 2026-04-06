// RUN: heir-opt --raise-ckks %s | FileCheck %s

// Test raising orion.linear_transform with constant diagonals to linalg.matvec.

!Z1099502714881_i64 = !mod_arith.int<1099502714881 : i64>
!Z1099503370241_i64 = !mod_arith.int<1099503370241 : i64>
!Z36028797019488257_i64 = !mod_arith.int<36028797019488257 : i64>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 1099511627776>
#key = #lwe.key<>
#modulus_chain = #lwe.modulus_chain<elements = <36028797019488257 : i64, 1099503370241 : i64, 1099502714881 : i64>, current = 2>
#ring_f64 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**8>>
!rns = !rns.rns<!Z36028797019488257_i64, !Z1099503370241_i64, !Z1099502714881_i64>
#ring_rns = #polynomial.ring<coefficientType = !rns, polynomialModulus = <1 + x**8>>
#ciphertext_space = #lwe.ciphertext_space<ring = #ring_rns, encryption_type = mix>
!ct = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space, key = #key, modulus_chain = #modulus_chain>

module attributes {scheme.ckks, ckks.schemeParam = #ckks.scheme_param<logN = 3, Q = [36028797019488257, 1099503370241, 1099502714881], P = [36028797019488257], logDefaultScale = 40>} {
  func.func @linear_transform(%ct: !ct) -> !ct {
    %diags = arith.constant dense<[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]> : tensor<2x4xf64>
    %ct_0 = orion.linear_transform %ct, %diags {block_col = 0 : i32, block_row = 0 : i32, bsgs_ratio = 0.0 : f64, diagonal_count = 2 : i64, diagonal_indices = array<i32: 0, 1>, orion_level = 2 : i64, slots = 4 : i64} : (!ct, tensor<2x4xf64>) -> !ct
    return %ct_0 : !ct
  }
}

// CHECK: module attributes
// CHECK-SAME: scheme.ckks
// CHECK-NOT: ckks.schemeParam
// CHECK: func.func @linear_transform(%[[ARG:.*]]: tensor<4xf64> {secret.secret})
// CHECK: arith.constant dense
// CHECK: linalg.matvec
// CHECK: return
