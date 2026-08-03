// Two chained composite ReLUs: the ~15-level sign chains exceed the
// bootstrap waterline budget, forcing the forward-level-sim to insert a
// bootstrap between them — the minimal bootstrap-in-the-loop composite case.
module {
  func.func @relu2_composite(%arg0 : tensor<16xf32> {secret.secret}) -> tensor<16xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<16xf32>
    %shift = arith.constant dense<1.000000e+00> : tensor<16xf32>
    %0 = arith.maximumf %arg0, %cst {domain_lower = -2.0733445882797241 : f64, domain_upper = 2.0503503084182739 : f64} : tensor<16xf32>
    %1 = arith.subf %0, %shift : tensor<16xf32>
    %2 = arith.maximumf %1, %cst {domain_lower = -1.5 : f64, domain_upper = 1.5 : f64} : tensor<16xf32>
    return %2 : tensor<16xf32>
  }
}
