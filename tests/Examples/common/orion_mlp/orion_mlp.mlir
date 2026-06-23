// This file contains the Orion MLP (from Orion): feedforward layers with
// Quad activations and BatchNorm folded into the preceding linear, the trained
// weights passed as function arguments. The export path is torch -> linalg.
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
module {
  func.func public @orion_mlp(%arg0: tensor<1x784xf32> {secret.secret}, %arg1: tensor<128x784xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128x128xf32>, %arg4: tensor<128xf32>, %arg5: tensor<10x128xf32>, %arg6: tensor<10xf32>) -> tensor<1x10xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<784x128xf32>
    %transposed = linalg.transpose ins(%arg1 : tensor<128x784xf32>) outs(%0 : tensor<784x128xf32>) permutation = [1, 0]
    %1 = tensor.empty() : tensor<1x128xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<1x128xf32>) -> tensor<1x128xf32>
    %3 = linalg.matmul ins(%arg0, %transposed : tensor<1x784xf32>, tensor<784x128xf32>) outs(%2 : tensor<1x128xf32>) -> tensor<1x128xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%3, %arg2 : tensor<1x128xf32>, tensor<128xf32>) outs(%1 : tensor<1x128xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %15 = arith.addf %in, %in_2 : f32
      linalg.yield %15 : f32
    } -> tensor<1x128xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %4 : tensor<1x128xf32>, tensor<1x128xf32>) outs(%1 : tensor<1x128xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %15 = arith.mulf %in, %in_2 : f32
      linalg.yield %15 : f32
    } -> tensor<1x128xf32>
    %6 = tensor.empty() : tensor<128x128xf32>
    %transposed_0 = linalg.transpose ins(%arg3 : tensor<128x128xf32>) outs(%6 : tensor<128x128xf32>) permutation = [1, 0]
    %7 = linalg.matmul ins(%5, %transposed_0 : tensor<1x128xf32>, tensor<128x128xf32>) outs(%2 : tensor<1x128xf32>) -> tensor<1x128xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%7, %arg4 : tensor<1x128xf32>, tensor<128xf32>) outs(%1 : tensor<1x128xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %15 = arith.addf %in, %in_2 : f32
      linalg.yield %15 : f32
    } -> tensor<1x128xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%8, %8 : tensor<1x128xf32>, tensor<1x128xf32>) outs(%1 : tensor<1x128xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %15 = arith.mulf %in, %in_2 : f32
      linalg.yield %15 : f32
    } -> tensor<1x128xf32>
    %10 = tensor.empty() : tensor<128x10xf32>
    %transposed_1 = linalg.transpose ins(%arg5 : tensor<10x128xf32>) outs(%10 : tensor<128x10xf32>) permutation = [1, 0]
    %11 = tensor.empty() : tensor<1x10xf32>
    %12 = linalg.fill ins(%cst : f32) outs(%11 : tensor<1x10xf32>) -> tensor<1x10xf32>
    %13 = linalg.matmul ins(%9, %transposed_1 : tensor<1x128xf32>, tensor<128x10xf32>) outs(%12 : tensor<1x10xf32>) -> tensor<1x10xf32>
    %14 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%13, %arg6 : tensor<1x10xf32>, tensor<10xf32>) outs(%11 : tensor<1x10xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %15 = arith.addf %in, %in_2 : f32
      linalg.yield %15 : f32
    } -> tensor<1x10xf32>
    return %14 : tensor<1x10xf32>
  }
}
