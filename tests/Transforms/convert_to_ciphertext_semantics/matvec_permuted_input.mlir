// RUN: heir-opt %s "--convert-to-ciphertext-semantics=ciphertext-size=8 use-lintrans-kernels=true" | FileCheck %s

// A public matrix can absorb a single-ciphertext input permutation. The
// resulting diagonal matrix below is checked element-for-element: each output
// slot consumes the slot holding its logical input column, and the usual squat
// post-shift still produces every row-major output replica.

#input_layout = #tensor_ext.layout<"{ [i] -> [ct, slot] : ct = 0 and (slot - 3i) mod 8 = 0 and 0 <= i <= 7 and 0 <= slot <= 7 }">
#matrix_layout = #tensor_ext.layout<"{ [row, col] -> [diag, slot] : (row - slot) mod 4 = 0 and (-col + diag + slot) mod 8 = 0 and 0 <= row <= 3 and 0 <= col <= 7 and 0 <= diag <= 3 and 0 <= slot <= 7 }">
#output_layout = #tensor_ext.layout<"{ [row] -> [ct, slot] : ct = 0 and (slot - row) mod 4 = 0 and 0 <= row <= 3 and 0 <= slot <= 7 }">
#kernel = #secret.kernel<name = "MatvecDiagonal", force = false>

// CHECK: func.func @matvec_permuted_input
// CHECK-NOT: tensor_ext.convert_layout
// CHECK: arith.constant dense<[
// CHECK-SAME: [1.000000e+00, 1.200000e+01, 2.300000e+01, 2.600000e+01, 5.000000e+00, 1.600000e+01, 1.900000e+01, 3.000000e+01],
// CHECK-SAME: [4.000000e+00, 1.500000e+01, 1.800000e+01, 2.900000e+01, 8.000000e+00, 1.100000e+01, 2.200000e+01, 2.500000e+01],
// CHECK-SAME: [7.000000e+00, 1.000000e+01, 2.100000e+01, 3.200000e+01, 3.000000e+00, 1.400000e+01, 1.700000e+01, 2.800000e+01],
// CHECK-SAME: [2.000000e+00, 1.300000e+01, 2.400000e+01, 2.700000e+01, 6.000000e+00, 9.000000e+00, 2.000000e+01, 3.100000e+01]
// CHECK-SAME: ]> : tensor<4x8xf32>
// CHECK: tensor_ext.rotate_and_reduce
// CHECK-SAME: tensor_ext.lintrans
// CHECK-NOT: linalg.matvec
func.func @matvec_permuted_input(
    %arg0: !secret.secret<tensor<8xf32>> {tensor_ext.layout = #input_layout})
    -> (!secret.secret<tensor<4xf32>> {tensor_ext.layout = #output_layout}) {
  %matrix = arith.constant dense<[
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
    [17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
    [25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0]
  ]> : tensor<4x8xf32>
  %bias = arith.constant dense<0.0> : tensor<4xf32>
  %packed_matrix = tensor_ext.assign_layout %matrix {
    layout = #matrix_layout,
    tensor_ext.layout = #matrix_layout
  } : tensor<4x8xf32>
  %packed_bias = tensor_ext.assign_layout %bias {
    layout = #output_layout,
    tensor_ext.layout = #output_layout
  } : tensor<4xf32>
  %0 = secret.generic(
      %arg0 : !secret.secret<tensor<8xf32>> {tensor_ext.layout = #input_layout}) {
  ^body(%input: tensor<8xf32>):
    %1 = linalg.matvec {
      heir.matvec_input_slots = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8,
                                            0, 3, 6, 1, 4, 7, 2, 5>,
      secret.kernel = #kernel,
      tensor_ext.layout = #output_layout
    } ins(%packed_matrix, %input : tensor<4x8xf32>, tensor<8xf32>)
      outs(%packed_bias : tensor<4xf32>) -> tensor<4xf32>
    secret.yield %1 : tensor<4xf32>
  } -> (!secret.secret<tensor<4xf32>> {tensor_ext.layout = #output_layout})
  return %0 : !secret.secret<tensor<4xf32>>
}
