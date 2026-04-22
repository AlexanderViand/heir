module {
  func.func @matvec(%arg0 : tensor<2xf32> {secret.secret}) -> tensor<2xf32> {
    %matrix = arith.constant dense<[[1.0, 0.0], [0.0, 1.0]]> : tensor<2x2xf32>
    %bias = arith.constant dense<[0.0, 0.0]> : tensor<2xf32>
    %0 = linalg.matvec ins(%matrix, %arg0 : tensor<2x2xf32>, tensor<2xf32>) outs(%bias : tensor<2xf32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }
}
