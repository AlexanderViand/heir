module {
  func.func @tiny_mlp(%arg0: tensor<4xf32> {secret.secret}) -> tensor<2xf32> {
    %w0 = arith.constant dense<[
      [0.25, -0.10, 0.20, 0.05],
      [-0.15, 0.30, 0.10, -0.20],
      [0.05, 0.10, -0.25, 0.20],
      [0.20, 0.15, 0.05, -0.10],
      [-0.10, 0.05, 0.30, 0.15],
      [0.15, -0.20, 0.10, 0.25],
      [0.10, 0.20, -0.15, 0.05],
      [-0.05, 0.25, 0.20, -0.15]
    ]> : tensor<8x4xf32>
    %zero0 = arith.constant dense<0.0> : tensor<8xf32>
    %hidden = linalg.matvec
        ins(%w0, %arg0 : tensor<8x4xf32>, tensor<4xf32>)
        outs(%zero0 : tensor<8xf32>) -> tensor<8xf32>
    %activated = polynomial.eval
        #polynomial.typed_chebyshev_polynomial<[0.0, 0.75, 0.0, 0.25]> :
          !polynomial.polynomial<ring=<coefficientType=f64>>,
        %hidden {domain_lower = -1.0 : f64, domain_upper = 1.0 : f64}
        : tensor<8xf32>
    %w1 = arith.constant dense<[
      [0.20, -0.10, 0.15, 0.05, -0.20, 0.10, 0.25, -0.05],
      [-0.15, 0.20, 0.05, -0.10, 0.15, 0.25, -0.05, 0.10]
    ]> : tensor<2x8xf32>
    %zero1 = arith.constant dense<0.0> : tensor<2xf32>
    %out = linalg.matvec
        ins(%w1, %activated : tensor<2x8xf32>, tensor<8xf32>)
        outs(%zero1 : tensor<2xf32>) -> tensor<2xf32>
    return %out : tensor<2xf32>
  }
}
