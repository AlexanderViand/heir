// RUN: heir-opt %s --convert-to-ciphertext-semantics=min-slot-count=4 | FileCheck %s

// CHECK: module
module attributes {
  backend.openfhe,
  backend.config_override = {has_kernel_linear_transform = true}
} {
  // CHECK: @main
  func.func @main(%arg0: !secret.secret<tensor<4xf32>> {tensor_ext.layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and slot = i0 and 0 <= i0 <= 3 and 0 <= slot <= 3 }">}) -> (!secret.secret<tensor<2xf32>> {tensor_ext.layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and slot = i0 and 0 <= i0 <= 1 and 0 <= slot <= 3 }">}) {
    %cst = arith.constant dense<0.0> : tensor<2xf32>
    %cst_mat = arith.constant dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>
    %0 = tensor_ext.assign_layout %cst_mat {
      layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : exists (e0: ct + slot - i1 - 4e0 = 0 and 0 <= ct <= 3) and slot - i0 = 0 and 0 <= i0 <= 1 and 0 <= i1 <= 3 and 0 <= slot <= 3 }">,
      tensor_ext.layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : exists (e0: ct + slot - i1 - 4e0 = 0 and 0 <= ct <= 3) and slot - i0 = 0 and 0 <= i0 <= 1 and 0 <= i1 <= 3 and 0 <= slot <= 3 }">
    } : tensor<2x4xf32>
    %1 = tensor_ext.assign_layout %cst {
      layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and slot = i0 and 0 <= i0 <= 1 and 0 <= slot <= 3 }">,
      tensor_ext.layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and slot = i0 and 0 <= i0 <= 1 and 0 <= slot <= 3 }">
    } : tensor<2xf32>
    %2 = secret.generic(%arg0 : !secret.secret<tensor<4xf32>> {tensor_ext.layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and slot = i0 and 0 <= i0 <= 3 and 0 <= slot <= 3 }">}) {
    ^body(%input: tensor<4xf32>):
      // CHECK: kernel.linear_transform
      // CHECK-SAME: diagonal_indices = array<i64: 0, 1, 2, 3>
      // CHECK-SAME: tensor<4x4xf32> -> tensor<1x4xf32>
      // Squat 2x4 packing requires one post-transform half-width reduction.
      // CHECK: tensor_ext.rotate
      // CHECK: arith.addf
      %3 = linalg.matvec {
        secret.kernel = #secret.kernel<name = "MatvecDiagonal", force = false>,
        tensor_ext.layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and slot = i0 and 0 <= i0 <= 1 and 0 <= slot <= 3 }">
      } ins(%0, %input : tensor<2x4xf32>, tensor<4xf32>) outs(%1 : tensor<2xf32>) -> tensor<2xf32>
      secret.yield %3 : tensor<2xf32>
    } -> (!secret.secret<tensor<2xf32>> {tensor_ext.layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and slot = i0 and 0 <= i0 <= 1 and 0 <= slot <= 3 }">})
    return %2 : !secret.secret<tensor<2xf32>>
  }
}
