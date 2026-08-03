// RUN: heir-opt --split-input-file --polynomial-approximation %s | FileCheck %s

// CHECK: @test_exp
func.func @test_exp(%x: f32) -> f32 {
  // CHECK: %[[SCALE:.*]] = arith.constant 2.500000e-01 : f32
  // CHECK: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK: %[[SCALED:.*]] = arith.mulf %{{.*}}, %[[SCALE]] : f32
  // CHECK: %[[V0:.*]] = arith.addf %[[SCALED]], %[[ONE]] : f32
  // CHECK: %[[V1:.*]] = arith.mulf %[[V0]], %[[V0]] : f32
  // CHECK: %[[V2:.*]] = arith.mulf %[[V1]], %[[V1]] : f32
  // CHECK: return %[[V2]] : f32
  %0 = math.exp %x {degree = 3 : i32, domain_lower = -1.0 : f64, domain_upper = 1.0 : f64} : f32
  return %0 : f32
}

// -----

// CHECK: @test_exp_tensor
func.func @test_exp_tensor(%x: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: %[[SCALE:.*]] = arith.constant dense<7.812500e-03> : tensor<4xf32>
  // CHECK: %[[ONE:.*]] = arith.constant dense<1.000000e+00> : tensor<4xf32>
  // CHECK: %[[SCALED:.*]] = arith.mulf %{{.*}}, %[[SCALE]] : tensor<4xf32>
  // CHECK: %[[V0:.*]] = arith.addf %[[SCALED]], %[[ONE]] : tensor<4xf32>
  // CHECK: %[[V1:.*]] = arith.mulf %[[V0]], %[[V0]] : tensor<4xf32>
  %0 = math.exp %x : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK: @test_domain
func.func @test_domain(%x: f32) -> f32 {
  // The calibrated domain [-1, 2] is materialized as an explicit affine rescale
  // onto Chebyshev's native [-1, 1]; the eval itself is then on the unit interval.
  // CHECK: arith.mulf
  // CHECK: arith.addf
  // CHECK: polynomial.eval
  // CHECK-SAME: domain_lower = -1.000000e+00
  // CHECK-SAME: domain_upper = 1.000000e+00
  %0 = math.exp %x {degree = 3 : i32, domain_lower = -1.0 : f64, domain_upper = 2.0 : f64} : f32
  return %0 : f32
}

// -----

// CHECK: @test_sin_default_params
func.func @test_sin_default_params(%x: f32) -> f32 {
  // CHECK: polynomial.eval
  // CHECK-SAME: [{{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}]
  %0 = math.sin %x : f32
  return %0 : f32
}

// -----

// CHECK: @test_maximumf
func.func @test_maximumf(%x: tensor<10xf32>) -> tensor<10xf32> {
  // CHECK: polynomial.eval
  // CHECK-NOT: arith.maximumf
  %c0 = arith.constant dense<0.0> : tensor<10xf32>
  %0 = arith.maximumf %x, %c0 : tensor<10xf32>
  return %0 : tensor<10xf32>
}

// -----

// CHECK: @test_maximumf_domain
func.func @test_maximumf_domain(%x: tensor<10xf32>) -> tensor<10xf32> {
  // Domain [-1, 2] normalized onto [-1, 1] via an explicit rescale before eval.
  // CHECK: arith.mulf
  // CHECK: arith.addf
  // CHECK: polynomial.eval
  // CHECK-SAME: domain_lower = -1.000000e+00
  // CHECK-SAME: domain_upper = 1.000000e+00
  // CHECK-NOT: arith.maximumf
  %c0 = arith.constant dense<0.0> : tensor<10xf32>
  %0 = arith.maximumf %x, %c0 {degree = 3 : i32, domain_lower = -1.0 : f64, domain_upper = 2.0 : f64}: tensor<10xf32>
  return %0 : tensor<10xf32>
}

// -----


// CHECK: @test_maximumf_ignore_not_splat
func.func @test_maximumf_ignore_not_splat(%x: tensor<10xf32>) -> tensor<10xf32> {
  // CHECK-NOT: polynomial.eval
  %c0 = arith.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]> : tensor<10xf32>
  %0 = arith.maximumf %x, %c0 : tensor<10xf32>
  return %0 : tensor<10xf32>
}

// -----

// CHECK: @test_maximumf_ignore_arg
func.func @test_maximumf_ignore_arg(%x: tensor<10xf32>, %y: tensor<10xf32>) -> tensor<10xf32> {
  // CHECK-NOT: polynomial.eval
  %0 = arith.maximumf %x, %y : tensor<10xf32>
  return %0 : tensor<10xf32>
}

// -----

// CHECK: @test_log_default_params
func.func @test_log_default_params(%x: f32) -> f32 {
  // Default positive domain [0.1, 2.0] is normalized onto [-1, 1].
  // CHECK: arith.mulf
  // CHECK: arith.addf
  // CHECK: polynomial.eval
  // CHECK-SAME: domain_lower = -1.000000e+00
  // CHECK-SAME: domain_upper = 1.000000e+00
  %0 = math.log %x : f32
  return %0 : f32
}

// -----

// CHECK: @test_sqrt_default_params
func.func @test_sqrt_default_params(%x: f32) -> f32 {
  // Default domain [0.0, 2.0]: the rescale factor is exactly 1.0 (no mulf), so
  // only the shift onto [-1, 1] is materialized.
  // CHECK: arith.addf
  // CHECK: polynomial.eval
  // CHECK-SAME: domain_lower = -1.000000e+00
  // CHECK-SAME: domain_upper = 1.000000e+00
  %0 = math.sqrt %x : f32
  return %0 : f32
}
