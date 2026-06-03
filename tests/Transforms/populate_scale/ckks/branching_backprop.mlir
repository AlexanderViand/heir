// RUN: heir-opt %s --populate-scale-ckks | FileCheck %s

// This test ensures `meet` is implemented on scale lattice, or else
// backpropagation through region-branching ops will not work properly.

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797019389953, 35184372121601], P = [36028797019488257], logDefaultScale = 45>, scheme.ckks} {
  // CHECK: func @test_scf_if_scale_mismatch_init(%{{.*}} !secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 35184372088832 : i64>})
  func.func @test_scf_if_scale_mismatch_init(%arg0: i1, %arg1: !secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 35184372088832>}) -> !secret.secret<f32> {
    %cst = arith.constant 7.0 : f32
    %1 = secret.generic(%arg1 : !secret.secret<f32>) {
    ^body(%arg1_val: f32):
      %0 = scf.if %arg0 -> (f32) {
        // CHECK: mgmt.init
        // CHECK-SAME: level = 0
        // CHECK-SAME: dimension = 3
        // CHECK-SAME: scale = 1237940039285380274899124224 : i92
        %1 = mgmt.init %cst {mgmt.mgmt = #mgmt.mgmt<level = 1>} : f32
        scf.yield %1 : f32
      } else {
        // CHECK: arith.mulf
        // CHECK-SAME: level = 0
        // CHECK-SAME: dimension = 3
        // CHECK-SAME: scale = 1237940039285380274899124224 : i92
        %1 = arith.mulf %arg1_val, %arg1_val {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : f32
        scf.yield %1 : f32
      }
      secret.yield %0 : f32
    } -> (!secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>})
    return %1 : !secret.secret<f32>
  }
}
