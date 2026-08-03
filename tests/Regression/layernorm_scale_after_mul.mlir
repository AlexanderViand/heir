// RUN: heir-opt %s --torch-linalg-to-ckks='enable-split-preprocessing=true ckks-add-plaintext-needs-runtime-scale=true preserve-poly-eval=true use-lintrans-kernels=true modulus-switch-after-mul=true' | FileCheck %s

// Rescale-after-multiply already returns every multiplication result to the
// canonical scale for its level. Running the rescale-before-only cross-mul-
// depth rewrite used to insert a phantom adjust_scale in this LayerNorm graph.
// That made scale analysis underdetermined and, when the barriers were removed,
// produced a ckks.sub whose operands had scales 0 and 45.

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
module {
  // CHECK: func.func @lnmlp
  func.func @lnmlp(
      %arg0: tensor<1x16xf32> {secret.secret},
      %bias4: tensor<4xf32>,
      %weight4x16: tensor<4x16xf32>,
      %bias16: tensor<16xf32>,
      %weight0: tensor<16x16xf32>,
      %weight1: tensor<16x16xf32>,
      %weight2: tensor<16x16xf32>,
      %meanBias: tensor<16xf32>) -> tensor<1x4xf32> {
    %zero = arith.constant 0.0 : f32
    %empty16x16 = tensor.empty() : tensor<16x16xf32>
    %weight0T = linalg.transpose ins(%weight0 : tensor<16x16xf32>) outs(%empty16x16 : tensor<16x16xf32>) permutation = [1, 0]
    %empty1x16 = tensor.empty() : tensor<1x16xf32>
    %zeros1x16 = linalg.fill ins(%zero : f32) outs(%empty1x16 : tensor<1x16xf32>) -> tensor<1x16xf32>
    %fc1 = linalg.matmul ins(%arg0, %weight0T : tensor<1x16xf32>, tensor<16x16xf32>) outs(%zeros1x16 : tensor<1x16xf32>) -> tensor<1x16xf32>
    %fc1Bias = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%fc1, %bias16 : tensor<1x16xf32>, tensor<16xf32>) outs(%empty1x16 : tensor<1x16xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %out: f32):
      %sum = arith.addf %lhs, %rhs : f32
      linalg.yield %sum : f32
    } -> tensor<1x16xf32>
    %weight1T = linalg.transpose ins(%weight1 : tensor<16x16xf32>) outs(%empty16x16 : tensor<16x16xf32>) permutation = [1, 0]
    %mean = linalg.matmul ins(%fc1Bias, %weight1T : tensor<1x16xf32>, tensor<16x16xf32>) outs(%zeros1x16 : tensor<1x16xf32>) -> tensor<1x16xf32>
    %centered = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%fc1Bias, %mean : tensor<1x16xf32>, tensor<1x16xf32>) outs(%empty1x16 : tensor<1x16xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %out: f32):
      %diff = arith.subf %lhs, %rhs : f32
      linalg.yield %diff : f32
    } -> tensor<1x16xf32>
    %squared = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%centered, %centered : tensor<1x16xf32>, tensor<1x16xf32>) outs(%empty1x16 : tensor<1x16xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %out: f32):
      %product = arith.mulf %lhs, %rhs : f32
      linalg.yield %product : f32
    } -> tensor<1x16xf32>
    %weight2T = linalg.transpose ins(%weight2 : tensor<16x16xf32>) outs(%empty16x16 : tensor<16x16xf32>) permutation = [1, 0]
    %variance = linalg.matmul ins(%squared, %weight2T : tensor<1x16xf32>, tensor<16x16xf32>) outs(%zeros1x16 : tensor<1x16xf32>) -> tensor<1x16xf32>
    %varianceBias = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%variance, %meanBias : tensor<1x16xf32>, tensor<16xf32>) outs(%empty1x16 : tensor<1x16xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %out: f32):
      %sum = arith.addf %lhs, %rhs : f32
      linalg.yield %sum : f32
    } -> tensor<1x16xf32>
    %inverseStddev = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%varianceBias : tensor<1x16xf32>) outs(%empty1x16 : tensor<1x16xf32>) attrs = {domain_lower = 0.1 : f64, domain_upper = 2.0 : f64} {
    ^bb0(%in: f32, %out: f32):
      %rsqrt = math.rsqrt %in : f32
      linalg.yield %rsqrt : f32
    } -> tensor<1x16xf32>
    %normalized = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%centered, %inverseStddev : tensor<1x16xf32>, tensor<1x16xf32>) outs(%empty1x16 : tensor<1x16xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %out: f32):
      %product = arith.mulf %lhs, %rhs : f32
      linalg.yield %product : f32
    } -> tensor<1x16xf32>
    %empty16x4 = tensor.empty() : tensor<16x4xf32>
    %weight4x16T = linalg.transpose ins(%weight4x16 : tensor<4x16xf32>) outs(%empty16x4 : tensor<16x4xf32>) permutation = [1, 0]
    %empty1x4 = tensor.empty() : tensor<1x4xf32>
    %zeros1x4 = linalg.fill ins(%zero : f32) outs(%empty1x4 : tensor<1x4xf32>) -> tensor<1x4xf32>
    %fc2 = linalg.matmul ins(%normalized, %weight4x16T : tensor<1x16xf32>, tensor<16x4xf32>) outs(%zeros1x4 : tensor<1x4xf32>) -> tensor<1x4xf32>
    %result = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%fc2, %bias4 : tensor<1x4xf32>, tensor<4xf32>) outs(%empty1x4 : tensor<1x4xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %out: f32):
      %sum = arith.addf %lhs, %rhs : f32
      linalg.yield %sum : f32
    } -> tensor<1x4xf32>
    return %result : tensor<1x4xf32>
  }
}

// CHECK: func.func @lnmlp__decrypt__result0
