// RUN: heir-opt --torch-linalg-to-ckks='ciphertext-degree=8 preserve-structured-ops=true' --scheme-to-lattigo %s | FileCheck %s

// Test that --preserve-structured-ops preserves polynomial.eval through
// management and converts it to orion.chebyshev -> lattigo.ckks.chebyshev.

func.func @test_relu(%arg0: tensor<8xf32> {secret.secret}) -> tensor<8xf32> {
  %0 = call @relu(%arg0) {domain_lower = -5.0, domain_upper = 5.0} : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}
func.func private @relu(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %cst = arith.constant dense<0.0> : tensor<f32>
  %0 = tensor.empty() : tensor<8xf32>
  %bcast = linalg.broadcast ins(%cst : tensor<f32>) outs(%0 : tensor<8xf32>) dimensions = [0]
  %1 = tensor.empty() : tensor<8xf32>
  %mapped = linalg.map { arith.maximumf } ins(%arg0, %bcast : tensor<8xf32>, tensor<8xf32>) outs(%1 : tensor<8xf32>)
  return %mapped : tensor<8xf32>
}

// CHECK: lattigo.ckks.chebyshev
// CHECK-NOT: polynomial.eval
