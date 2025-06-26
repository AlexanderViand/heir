func.func @matmul(%2 : tensor<16xf32> {secret.secret}, %3 : tensor<16xf32>, %7 : tensor<16xf32>, %10 : tensor<16xf32>, %12 : tensor<16xf32>) -> tensor<16xf32> {
  %c5 = arith.constant 4 : index
  %c15 = arith.constant 8 : index
  %1 = arith.mulf %2, %3 : tensor<16xf32>
  %4 = tensor_ext.rotate %2, %c5 : tensor<16xf32>, index
  %6 = arith.mulf %4, %7 : tensor<16xf32>
  %8 = arith.addf %1, %6 : tensor<16xf32>
  %9 = arith.mulf %2, %10 : tensor<16xf32>
  %11 = arith.mulf %4, %12 : tensor<16xf32>
  %13 = arith.addf %9, %11 : tensor<16xf32>
  %14 = tensor_ext.rotate %13, %c15 : tensor<16xf32>, index
  %16 = arith.addf %8, %14 : tensor<16xf32>
  return %16 : tensor<16xf32>
}
