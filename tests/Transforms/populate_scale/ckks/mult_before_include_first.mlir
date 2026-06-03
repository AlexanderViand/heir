// RUN: heir-opt %s --split-input-file --secret-insert-mgmt-ckks=before-mul-include-first-mul --populate-scale-ckks=before-mul-include-first-mul | FileCheck %s

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 14, Q = [36028797019389953, 35184372121601, 35184372744193, 35184373006337], P = [36028797019488257, 36028797020209153], logDefaultScale = 45>, scheme.ckks} {
  // CHECK: func @mult
  func.func @mult(%arg0: !secret.secret<f32>) -> !secret.secret<f32> {
    // `before-mul-include-first-mul` seeds the exact scale at Delta^2.
    // CHECK: secret.generic
    // CHECK-SAME: level = 3
    // CHECK-SAME: scale = 1237940039285380274899124224 : i92
    %0 = secret.generic(%arg0 : !secret.secret<f32>) {
    ^body(%input0: f32):
      %1 = arith.mulf %input0, %input0 : f32
      %2 = arith.addf %1, %1 : f32
      %3 = arith.mulf %2, %2 : f32
      secret.yield %3 : f32
    // CHECK: secret.yield
    // CHECK: ->
    // CHECK-SAME: level = 0
    // CHECK-SAME: scale = 35184372088832 : i64
    } -> !secret.secret<f32>
    return %0 : !secret.secret<f32>
  }
}

// -----

// MatchCrossLevel

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 14, Q = [36028797019389953, 35184372121601, 35184372744193], P = [36028797019488257, 36028797020209153], logDefaultScale = 45>, scheme.ckks} {
  // CHECK: func @mul
  func.func @mul(%arg0: !secret.secret<f32>) -> !secret.secret<f32> {
    // adjust_scale needs plaintext input encoded in specified scale
    // CHECK: %cst = arith.constant 1.000000e+00
    // CHECK: %[[INIT:.*]] = mgmt.init %cst
    // CHECK-SAME: level = 2
    // CHECK-SAME: scale = 35184372088832 : i64

    // CHECK: secret.generic
    // CHECK-SAME: level = 2
    // CHECK-SAME: scale = 1237940039285380274899124224 : i92
    %0 = secret.generic(%arg0 : !secret.secret<f32>) {
    // CHECK: ^body(%[[INPUT0:.*]]: f32):
    ^body(%input0: f32):
      // CHECK: %[[v2:.*]] = mgmt.modreduce %[[INPUT0]]
      // CHECK-NEXT: %[[v3:.*]] = arith.mulf %[[v2]], %[[v2]]
      // CHECK-NEXT: %[[v4:.*]] = mgmt.relinearize %[[v3]]
      %1 = arith.mulf %input0, %input0 : f32
      // Under the nominal-bucket policy, adjust_scale targets Delta^2 exactly,
      // so the rescaled peer lands on the same nominal scale as the mul path.
      // CHECK-NEXT: %[[v5:.*]] = arith.mulf %[[INPUT0]], %[[INIT]]
      // CHECK-NEXT: %[[v6:.*]] = mgmt.modreduce %[[v5]]
      // CHECK-SAME: scale = 1237940039285380274899124224 : i92
      // CHECK-NEXT: %[[v7:.*]] = arith.addf %[[v4]], %[[v6]]
      %2 = arith.addf %1, %input0 : f32
      // CHECK-NEXT: %[[v8:.*]] = mgmt.modreduce %[[v7]]
      // CHECK-NEXT: secret.yield %[[v8]]
      secret.yield %2 : f32
    // CHECK: ->
    // CHECK-SAME: level = 0
    // CHECK-SAME: scale = 35184372088832 : i64
    } -> !secret.secret<f32>
    return %0 : !secret.secret<f32>
  }
}
