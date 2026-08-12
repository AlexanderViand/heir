module {
  func.func @chebyshev_cube(
      %arg0: tensor<8xf32> {secret.secret}) -> tensor<8xf32> {
    %0 = polynomial.eval
        #polynomial.typed_chebyshev_polynomial<[
          0.0, 0.625, 0.0, 0.3125, 0.0, 0.0625
        ]> :
          !polynomial.polynomial<ring=<coefficientType=f64>>,
        %arg0 {domain_lower = -1.0 : f64, domain_upper = 1.0 : f64}
        : tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}
