// RUN: heir-opt --torch-linalg-to-ckks %s | FileCheck %s

// Regression test: an nn.Linear(*, 1) (i.e. tensor<*x1> output) used to make
// --torch-linalg-to-ckks emit a tensor.extract with zero indices on a rank-2
// ciphertext tensor. ISL's AST builder drops domain dims from the statement
// call when ALL domain dims are forced to constants by the relation — which
// happens exactly when the data tensor has a single element — so the unpack
// loop body received fewer index exprs than the ciphertext rank.

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @main(%arg0: tensor<1x2xf32> {secret.secret}) -> tensor<1x1xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant dense<[[1.0, 2.0]]> : tensor<1x2xf32>
    %cst_1 = arith.constant dense<0.124900162> : tensor<1xf32>
    %0 = tensor.empty() : tensor<2x1xf32>
    %transposed = linalg.transpose ins(%cst_0 : tensor<1x2xf32>) outs(%0 : tensor<2x1xf32>) permutation = [1, 0]
    %1 = tensor.empty() : tensor<1x1xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<1x1xf32>) -> tensor<1x1xf32>
    %3 = linalg.matmul ins(%arg0, %transposed : tensor<1x2xf32>, tensor<2x1xf32>) outs(%2 : tensor<1x1xf32>) -> tensor<1x1xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%3, %cst_1 : tensor<1x1xf32>, tensor<1xf32>) outs(%1 : tensor<1x1xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %5 = arith.addf %in, %in_2 : f32
      linalg.yield %5 : f32
    } -> tensor<1x1xf32>
    return %4 : tensor<1x1xf32>
  }
}

// CHECK: func.func @main__decrypt__result0
// CHECK: tensor.extract %{{[^[]+}}[%{{[^,]+}}, %{{[^]]+}}] : tensor<{{[0-9]+}}x{{[0-9]+}}xf32>
// CHECK: tensor.from_elements %{{.*}} : tensor<1x1xf32>
