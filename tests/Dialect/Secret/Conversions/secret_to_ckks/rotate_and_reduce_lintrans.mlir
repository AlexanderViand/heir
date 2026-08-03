// RUN: heir-opt --mlir-print-local-scope --secret-to-ckks %s | FileCheck %s

// A lintrans-marked tensor_ext.rotate_and_reduce (kept compact by
// convert-to-ciphertext-semantics under use-lintrans-kernels) lowers to
// orion.linear_transform: dense diagonal indices, orion_level = the input
// ciphertext's level, slots = the diagonal row width.

!ef = !secret.secret<tensor<1024xf32>>

#mgmt = #mgmt.mgmt<level = 2, dimension = 2>
#mgmt_sq = #mgmt.mgmt<level = 2, dimension = 2, scale = 90>

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 14, Q = [36028797019389953, 35184372121601, 35184372744193, 35184373006337, 35184373989377, 35184374874113], P = [36028797019488257, 36028797020209153], logDefaultScale = 45>} {
  // CHECK: func @test_lintrans
  // CHECK-SAME: %[[arg0:.*]]: !lwe.lwe_ciphertext
  // CHECK-SAME: %[[arg1:.*]]: tensor<32x1024xf32>
  func.func @test_lintrans(%arg0 : !ef {mgmt.mgmt = #mgmt}, %arg1 : tensor<32x1024xf32>) -> (!ef {mgmt.mgmt = #mgmt_sq}) {
    %arg1_attr = mgmt.init %arg1 {mgmt.mgmt = #mgmt} : tensor<32x1024xf32>
    %0 = secret.generic(%arg0 :  !ef) {
    // CHECK: orion.linear_transform %[[arg0]], %[[arg1]]
    // CHECK-SAME: bsgs_ratio = 2.0
    // CHECK-SAME: diagonal_indices = array<i32: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31>
    // CHECK-SAME: orion_level = 2 : i32
    // CHECK-SAME: slots = 1024 : i32
    // CHECK-NOT: tensor_ext.rotate_and_reduce
      ^bb0(%ARG0 : tensor<1024xf32>):
        %1 = tensor_ext.rotate_and_reduce %ARG0, %arg1_attr {period = 1 : index, steps = 32 : index, reduceOp = "arith.addf", tensor_ext.lintrans} : (tensor<1024xf32>, tensor<32x1024xf32>) -> tensor<1024xf32>
        secret.yield %1 : tensor<1024xf32>
    } -> (!ef {mgmt.mgmt = #mgmt_sq})
    return %0 : !ef
  }
}
