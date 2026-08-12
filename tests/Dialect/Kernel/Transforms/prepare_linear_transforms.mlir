// RUN: heir-opt --split-input-file --prepare-linear-transforms %s | FileCheck %s

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 45>
#key = #lwe.key<>
#modulus_chain = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64>, current = 0>
#modulus_chain_L1 = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64>, current = 1>
#ring_f64_1_x1024 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_L0 = !rns.rns<!mod_arith.int<36028797018652673 : i64>>
!rns_L1 = !rns.rns<!mod_arith.int<36028797018652673 : i64>, !mod_arith.int<35184372121601 : i64>>
#ring_rns_L0_1_x1024 = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**1024>>
#ring_rns_L1_1_x1024 = #polynomial.ring<coefficientType = !rns_L1, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L0 = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024, encryption_type = mix>
#ciphertext_space_L1 = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024, encryption_type = mix>
!ct = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L0, key = #key, modulus_chain = #modulus_chain>
!ct_L1 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L1, key = #key, modulus_chain = #modulus_chain_L1>

// The ciphertext's chain has current 0, so it sits at level 0; its ring has
// degree 1024 with inverse-canonical encoding, so 512 slots.

// CHECK: @split
// CHECK: %[[LT:.*]] = kernel.prepare_linear_transform %{{.*}} {diagonal_indices = array<i64: 0, 2>, source_row_indices = array<i64: 1, 3>} : tensor<4x512xf64> -> <level = 0, slots = 512, log_bsgs_ratio = 0>
// CHECK: %[[OUT:.*]] = kernel.apply_linear_transform %{{.*}}, %[[LT]] {diagonal_indices = array<i64: 0, 2>, diagonal_width = 512 : i64, kernel.test} : {{.*}}<level = 0, slots = 512, log_bsgs_ratio = 0>{{.*}}
// CHECK-NOT: kernel.linear_transform
// CHECK: return %[[OUT]]
module attributes {backend.cheddar, scheme.ckks} {
  func.func @split(%ct: !ct) -> !ct {
    %diagonals = arith.constant dense<1.0> : tensor<4x512xf64>
    %0 = kernel.linear_transform %ct, %diagonals {diagonal_indices = array<i64: 0, 2>, kernel.test, source_row_indices = array<i64: 1, 3>} : !ct, tensor<4x512xf64> -> !ct
    return %0 : !ct
  }

  // CHECK: @single_diagonal
  // CHECK: %[[ROW:.*]] = tensor.extract_slice
  // CHECK: %[[ROTATED:.*]] = ckks.rotate %{{.*}} {static_shift = 6 : index}
  // CHECK: %[[PT:.*]] = lwe.rlwe_encode %[[ROW]] {encoding = #inverse_canonical_encoding, level = 1 : i64, ring = #ring_f64_1_x1024, scale = 45 : i64}
  // CHECK: %[[PRODUCT:.*]] = ckks.mul_plain %[[ROTATED]], %[[PT]]
  // CHECK: %[[OUT:.*]] = ckks.rescale %[[PRODUCT]]
  // CHECK-NOT: kernel.prepare_linear_transform
  // CHECK-NOT: kernel.linear_transform
  // CHECK: return %[[OUT]]
  func.func @single_diagonal(%ct: tensor<1x!ct_L1>, %diagonals: tensor<1x512xf64>) -> tensor<1x!ct> {
    %0 = kernel.linear_transform %ct, %diagonals {diagonal_indices = array<i64: 6>} : tensor<1x!ct_L1>, tensor<1x512xf64> -> tensor<1x!ct>
    return %0 : tensor<1x!ct>
  }
}

// -----

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 45>
#key = #lwe.key<>
#modulus_chain = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64>, current = 0>
#ring_f64_1_x1024 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_L0 = !rns.rns<!mod_arith.int<36028797018652673 : i64>>
#ring_rns_L0_1_x1024 = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L0 = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024, encryption_type = mix>
!ct = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L0, key = #key, modulus_chain = #modulus_chain>

// A target that does not declare has_prepared_linear_transform keeps the
// sugar form.

// CHECK: @keeps_sugar
// CHECK: kernel.linear_transform
// CHECK-NOT: kernel.prepare_linear_transform
module attributes {backend.openfhe, scheme.ckks} {
  func.func @keeps_sugar(%ct: !ct) -> !ct {
    %diagonals = arith.constant dense<1.0> : tensor<2x512xf64>
    %0 = kernel.linear_transform %ct, %diagonals {diagonal_indices = array<i64: 0, 1>} : !ct, tensor<2x512xf64> -> !ct
    return %0 : !ct
  }
}
