// RUN: heir-opt %s --secret-insert-mgmt-ckks=before-mul-include-first-mul '--populate-scale-ckks=before-mul-include-first-mul scale-policy=precise' | FileCheck %s

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 14, Q = [36028797019389953, 35184372121601, 35184372744193], P = [36028797019488257, 36028797020209153], logDefaultScale = 45>, scheme.ckks} {
  // CHECK: func @mul
  // CHECK: %[[CST:.*]] = arith.constant 1.000000e+00 : f32
  func.func @mul(%arg0: !secret.secret<f32>) -> !secret.secret<f32> {
    // CHECK: secret.generic
    // CHECK-SAME: level = 3
    %0 = secret.generic(%arg0 : !secret.secret<f32>) {
    // CHECK: ^body(%[[INPUT0:.*]]: f32):
    ^body(%input0: f32):
      // CHECK: %[[REDUCE_INPUT:.*]] = mgmt.modreduce %[[INPUT0]]
      // CHECK-NEXT: %[[MUL:.*]] = arith.mulf %[[REDUCE_INPUT]], %[[REDUCE_INPUT]]
      // CHECK-NEXT: %[[RELIN:.*]] = mgmt.relinearize %[[MUL]]
      // CHECK-NEXT: %[[INIT:.*]] = mgmt.init %[[CST]]
      %1 = arith.mulf %input0, %input0 : f32
      // CHECK-NEXT: %[[ADJUST:.*]] = arith.mulf %[[INPUT0]], %[[INIT]]
      // CHECK-NEXT: %[[ADJUST_REDUCED_0:.*]] = mgmt.modreduce %[[ADJUST]]
      // CHECK-NEXT: %[[ADJUST_REDUCED_1:.*]] = mgmt.modreduce %[[ADJUST_REDUCED_0]]
      // CHECK-NEXT: %[[MUL_REDUCED_1:.*]] = mgmt.modreduce %[[RELIN]]
      // CHECK-NEXT: %[[ADD:.*]] = arith.addf %[[MUL_REDUCED_1]], %[[ADJUST_REDUCED_1]]
      %2 = arith.addf %1, %input0 : f32
      // CHECK-NEXT: %[[OUT:.*]] = mgmt.modreduce %[[ADD]]
      // CHECK-NEXT: secret.yield %[[OUT]]
      secret.yield %2 : f32
    } -> !secret.secret<f32>
    return %0 : !secret.secret<f32>
  }
}
