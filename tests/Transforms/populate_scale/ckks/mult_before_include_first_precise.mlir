// RUN: heir-opt %s --secret-insert-mgmt-ckks=before-mul-include-first-mul --resolve-reconcile-ckks --resolve-scale-ckks-bmph20=before-mul-include-first-mul | FileCheck %s

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 14, Q = [36028797019389953, 35184372121601, 35184372744193], P = [36028797019488257, 36028797020209153], logDefaultScale = 45>, scheme.ckks} {
  // CHECK: func.func @mul
  func.func @mul(%arg0: !secret.secret<f32>) -> !secret.secret<f32> {
    // CHECK: secret.generic
    // CHECK-SAME: level = 3
    %0 = secret.generic(%arg0 : !secret.secret<f32>) {
    // CHECK: ^body(%[[INPUT0:.*]]: f32):
    ^body(%input0: f32):
      // CHECK: %[[REDUCE_INPUT:.*]] = mgmt.modreduce %[[INPUT0]]
      // CHECK-NEXT: %[[MUL:.*]] = arith.mulf %[[REDUCE_INPUT]], %[[REDUCE_INPUT]]
      // CHECK-NEXT: %[[RELIN:.*]] = mgmt.relinearize %[[MUL]]
      %1 = arith.mulf %input0, %input0 : f32
      // BMPH20-style precise refinement is free to lower the meeting point
      // further than the coarse add level.
      // CHECK-NEXT: %[[INIT:.*]] = mgmt.init
      // CHECK-NEXT: %[[INPUT_MUL:.*]] = arith.mulf %[[INPUT0]], %[[INIT]]
      // CHECK-NEXT: %[[INPUT_REDUCED_0:.*]] = mgmt.modreduce %[[INPUT_MUL]]
      // CHECK-NEXT: %[[INPUT_REDUCED_1:.*]] = mgmt.modreduce %[[INPUT_REDUCED_0]]
      // CHECK-NEXT: %[[MUL_REDUCED_1:.*]] = mgmt.modreduce %[[RELIN]]
      // CHECK-NEXT: %[[ADD:.*]] = arith.addf %[[MUL_REDUCED_1]], %[[INPUT_REDUCED_1]]
      // CHECK-SAME: level = 1
      %2 = arith.addf %1, %input0 : f32
      // CHECK-NEXT: %[[OUT:.*]] = mgmt.modreduce %[[ADD]]
      // CHECK-NEXT: secret.yield %[[OUT]]
      secret.yield %2 : f32
    } -> !secret.secret<f32>
    return %0 : !secret.secret<f32>
  }
}
