// RUN: heir-opt --mlir-print-local-scope "--secret-to-ckks=poly-mod-degree=16" %s | FileCheck %s

// A lintrans-marked rotate_and_reduce over a mostly-zero constant diagonal
// tensor (a conv's expanded Toeplitz matrix) carries the nonzero rows'
// offsets in a tensor_ext.diagonal_indices attr. SecretToCKKS compacts the
// constant down to those rows and forwards the offsets as
// orion.linear_transform's diagonal_indices (row i of the compacted tensor is
// diagonal diagonalIndices[i]), so the backend kernel neither encodes nor
// rotates for the zero diagonals.

!ef = !secret.secret<tensor<16xf32>>

#mgmt = #mgmt.mgmt<level = 2, dimension = 2>
#mgmt_sq = #mgmt.mgmt<level = 2, dimension = 2, scale = 90>

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 14, Q = [36028797019389953, 35184372121601, 35184372744193, 35184373006337, 35184373989377, 35184374874113], P = [36028797019488257, 36028797020209153], logDefaultScale = 45>} {
  // CHECK: func @test_lintrans_sparse
  // CHECK-SAME: %[[arg0:.*]]: !lwe.lwe_ciphertext
  func.func @test_lintrans_sparse(%arg0 : !ef {mgmt.mgmt = #mgmt}) -> (!ef {mgmt.mgmt = #mgmt_sq}) {
    // Rows 1 and 3 are all-zero; rows 0 (ones) and 2 (twos) survive in the
    // compacted constant.
    // CHECK: %[[compact:.*]] = arith.constant dense<{{\[}}[1.000000e+00, {{.*}}, 2.000000e+00]]> : tensor<2x16xf32>
    %diags = arith.constant dense<[
      [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ]> : tensor<4x16xf32>
    %diags_init = mgmt.init %diags {mgmt.mgmt = #mgmt} : tensor<4x16xf32>
    %0 = secret.generic(%arg0 :  !ef) {
    // CHECK: orion.linear_transform %[[arg0]], %[[compact]]
    // CHECK-SAME: diagonal_indices = array<i32: 0, 2>
    // CHECK-SAME: orion_level = 2 : i32
    // CHECK-SAME: slots = 16 : i32
    // CHECK-NOT: tensor_ext.rotate_and_reduce
      ^bb0(%ARG0 : tensor<16xf32>):
        %1 = tensor_ext.rotate_and_reduce %ARG0, %diags_init {period = 1 : index, steps = 4 : index, reduceOp = "arith.addf", tensor_ext.lintrans, tensor_ext.diagonal_indices = array<i32: 0, 2>} : (tensor<16xf32>, tensor<4x16xf32>) -> tensor<16xf32>
        secret.yield %1 : tensor<16xf32>
    } -> (!ef {mgmt.mgmt = #mgmt_sq})
    return %0 : !ef
  }
}
