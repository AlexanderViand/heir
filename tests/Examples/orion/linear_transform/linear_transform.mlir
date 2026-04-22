// RUN: heir-opt --annotate-orion --ckks-to-lwe --lwe-to-lattigo %s | FileCheck %s

// CHECK: @linear_transform
// CHECK-NOT: orion.linear_transform
// CHECK: lattigo.ckks.linear_transform

!Z536903681_i64 = !mod_arith.int<536903681 : i64>
!Z66994177_i64 = !mod_arith.int<66994177 : i64>
!Z67043329_i64 = !mod_arith.int<67043329 : i64>
!Z67239937_i64 = !mod_arith.int<67239937 : i64>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 67108864>
#key = #lwe.key<>
#modulus_chain_L3_C3 = #lwe.modulus_chain<elements = <536903681 : i64, 67043329 : i64, 66994177 : i64, 67239937 : i64>, current = 3>
#ring_f64_1_x8192 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**8192>>
!rns_L3 = !rns.rns<!Z536903681_i64, !Z67043329_i64, !Z66994177_i64, !Z67239937_i64>
#ring_rns_L3_1_x8192 = #polynomial.ring<coefficientType = !rns_L3, polynomialModulus = <1 + x**8192>>
#ciphertext_space_L3 = #lwe.ciphertext_space<ring = #ring_rns_L3_1_x8192, encryption_type = mix>
!ct_L3 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x8192, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L3, key = #key, modulus_chain = #modulus_chain_L3_C3>
module attributes {scheme.ckks, ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [536903681, 67043329, 66994177, 67239937], P = [536952833, 536690689], logDefaultScale = 26>} {
  func.func @linear_transform(%ct: !ct_L3, %arg0: tensor<2x4096xf64> {orion.block_col = 0 : i64, orion.block_row = 0 : i64, orion.layer_name = "fc1", orion.layer_role = "weights", orion.level = 3 : i64}) -> !ct_L3 {
    %ct_0 = orion.linear_transform %ct, %arg0 {block_col = 0 : i32, block_row = 0 : i32, bsgs_ratio = 2.000000e+00 : f64, diagonal_count = 2 : i32, orion_level = 3 : i32, slots = 4096 : i32, diagonal_indices = array<i32: 0, 1>} : (!ct_L3, tensor<2x4096xf64>) -> !ct_L3
    return %ct_0 : !ct_L3
  }
}
