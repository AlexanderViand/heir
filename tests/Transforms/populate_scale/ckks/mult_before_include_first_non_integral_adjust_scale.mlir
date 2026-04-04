// RUN: heir-opt %s --secret-insert-mgmt-ckks=before-mul-include-first-mul --populate-scale-ckks=before-mul-include-first-mul | FileCheck %s

// This case used to require non-integral adjustment under exact dropped-prime
// arithmetic. Under the nominal-bucket policy, the adjust-scale target is the
// exact Delta^2 bucket, so the materialized plaintext scale is again Delta.

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 14, Q = [36028797019389953, 35184372121601, 35184372744193], P = [36028797019488257, 36028797020209153], logDefaultScale = 45>, scheme.ckks} {
  func.func @mul(%arg0: !secret.secret<f32>) -> !secret.secret<f32> {
    // CHECK: %[[INIT:.*]] = mgmt.init
    %0 = secret.generic(%arg0 : !secret.secret<f32>) {
    ^body(%input0: f32):
      // CHECK: %[[REDUCE:.*]] = mgmt.modreduce %input0
      // CHECK-NEXT: %[[MUL:.*]] = arith.mulf %[[REDUCE]], %[[REDUCE]]
      // CHECK-NEXT: %[[RELIN:.*]] = mgmt.relinearize %[[MUL]]
      %1 = arith.mulf %input0, %input0 : f32
      // Under the nominal policy the resolver reconstructs the historical
      // one-sided mul_plain path so the add still lands in the Delta^2 bucket.
      // CHECK-NEXT: %[[ADJUST_MUL:.*]] = arith.mulf %input0, %[[INIT]]
      // CHECK-NEXT: %[[ADJUST_DROP:.*]] = mgmt.modreduce %[[ADJUST_MUL]]
      // CHECK-NEXT: %[[ADD:.*]] = arith.addf %[[RELIN]], %[[ADJUST_DROP]]
      %2 = arith.addf %1, %input0 : f32
      // CHECK-NEXT: %[[OUT:.*]] = mgmt.modreduce %[[ADD]]
      // CHECK-NEXT: secret.yield %[[OUT]]
      secret.yield %2 : f32
    } -> !secret.secret<f32>
    return %0 : !secret.secret<f32>
  }
}
