// RUN: heir-opt --lwe-to-cheddar %s | FileCheck %s
// RUN: heir-opt --lwe-to-cheddar=enable-min-ks=false %s | FileCheck %s --check-prefix=NO-MIN-KS

#enc = #lwe.inverse_canonical_encoding<scaling_factor = 45>
#key = #lwe.key<>
#chain_in = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64>, current = 1>
#chain_out = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64>, current = 0>
#rf = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_in = !rns.rns<!mod_arith.int<36028797018652673 : i64>, !mod_arith.int<35184372121601 : i64>>
!rns_out = !rns.rns<!mod_arith.int<36028797018652673 : i64>>
#rr_in = #polynomial.ring<coefficientType = !rns_in, polynomialModulus = <1 + x**1024>>
#rr_out = #polynomial.ring<coefficientType = !rns_out, polynomialModulus = <1 + x**1024>>
#cs_in = #lwe.ciphertext_space<ring = #rr_in, encryption_type = mix>
#cs_out = #lwe.ciphertext_space<ring = #rr_out, encryption_type = mix>
!ct_in = !lwe.lwe_ciphertext<plaintext_space = <ring = #rf, encoding = #enc>, ciphertext_space = #cs_in, key = #key, modulus_chain = #chain_in>
!ct_out = !lwe.lwe_ciphertext<plaintext_space = <ring = #rf, encoding = #enc>, ciphertext_space = #cs_out, key = #key, modulus_chain = #chain_out>
!prepared = !kernel.prepared_linear_transform<level = 1, slots = 512, log_bsgs_ratio = 0>

module attributes {backend.cheddar, ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797018652673, 35184372121601], P = [1152921504606994433], logDefaultScale = 45, encryptionTechnique = extended>, scheme.ckks} {
  // CHECK: func.func @linear_transform
  // CHECK: arith.constant dense<
  // CHECK: cheddar.linear_transform
  // CHECK-SAME: bs = 4 : i64
  // CHECK-SAME: diagonal_indices = array<i32: 0, 1, 3>
  // CHECK-SAME: gs = 1 : i64
  // CHECK-SAME: level = 1 : i64
  func.func @linear_transform(%ct: !ct_in) -> !ct_out {
    %diagonals = arith.constant dense<1.0> : tensor<3x8xf64>
    %0 = kernel.linear_transform %ct, %diagonals {diagonal_indices = array<i64: 0, 1, 3>} : !ct_in, tensor<3x8xf64> -> !ct_out
    return %0 : !ct_out
  }

  // Complete baby- and giant-step progressions select scale-snu's minimum-key
  // switch path and plan only the two progression-stride keys.
  // CHECK: func.func @linear_transform_min_ks
  // CHECK: cheddar.linear_transform
  // CHECK-SAME: bs = 4 : i64
  // CHECK-SAME: diagonal_indices = array<i32: 0, 1, 2, 3, 4, 5, 6, 7>
  // CHECK-SAME: gs = 2 : i64
  // CHECK-SAME: level = 1 : i64
  // CHECK-SAME: min_ks = true
  // NO-MIN-KS: func.func @linear_transform_min_ks
  // NO-MIN-KS: cheddar.linear_transform
  // NO-MIN-KS-NOT: min_ks = true
  // NO-MIN-KS: func.func @prepare_linear_transform
  func.func @linear_transform_min_ks(%ct: !ct_in) -> !ct_out {
    %diagonals = arith.constant dense<1.0> : tensor<8x8xf64>
    %0 = kernel.linear_transform %ct, %diagonals {diagonal_indices = array<i64: 0, 1, 2, 3, 4, 5, 6, 7>} : !ct_in, tensor<8x8xf64> -> !ct_out
    return %0 : !ct_out
  }

  // CHECK: func.func @prepare_linear_transform(
  // CHECK-SAME: !context
  // CHECK: cheddar.prepare_linear_transform
  // CHECK-SAME: bs = 4 : i64
  // CHECK-SAME: diagonal_indices = array<i32: 0, 1, 3>
  // CHECK-SAME: gs = 1 : i64
  // CHECK-SAME: level = 1 : i64
  // CHECK-SAME: source_row_indices = array<i32: 0, 2, 4>
  func.func @prepare_linear_transform(%diagonals: tensor<5x8xf64>) -> !prepared {
    %0 = kernel.prepare_linear_transform %diagonals {diagonal_indices = array<i64: 0, 1, 3>, source_row_indices = array<i64: 0, 2, 4>} : tensor<5x8xf64> -> !prepared
    return %0 : !prepared
  }

  // CHECK: func.func @apply_linear_transform(
  // CHECK: cheddar.apply_prepared_linear_transform
  // CHECK-SAME: min_ks = true
  func.func @apply_linear_transform(%ct: !ct_in, %prepared: !prepared) -> !ct_out {
    %0 = kernel.apply_linear_transform %ct, %prepared {diagonal_indices = array<i64: 0, 1, 2, 3, 4, 5, 6, 7>, diagonal_width = 8 : i64} : !ct_in, !prepared -> !ct_out
    return %0 : !ct_out
  }
}
