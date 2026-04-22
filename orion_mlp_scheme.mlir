module attributes {heir.min_level_hint = 5 : i64, scheme.ckks} {
  func.func @mlp(%arg0: tensor<4096xf64> {secret.secret}, %arg1: tensor<128x4096xf64>, %arg2: tensor<4096xf64>, %arg3: tensor<128x4096xf64>, %arg4: tensor<4096xf64>, %arg5: tensor<137x4096xf64>, %arg6: tensor<4096xf64>) -> tensor<4096xf64> {
    %0 = tensor_ext.diagonal_matvec %arg0, %arg1 {diagonal_indices = array<i32: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127>, slots = 4096 : i64} : (tensor<4096xf64>, tensor<128x4096xf64>) -> tensor<4096xf64>
    %c2048 = arith.constant 2048 : index
    %1 = tensor_ext.rotate %0, %c2048 : tensor<4096xf64>, index
    %2 = arith.addf %1, %0 : tensor<4096xf64>
    %c1024 = arith.constant 1024 : index
    %3 = tensor_ext.rotate %2, %c1024 : tensor<4096xf64>, index
    %4 = arith.addf %3, %2 : tensor<4096xf64>
    %c512 = arith.constant 512 : index
    %5 = tensor_ext.rotate %4, %c512 : tensor<4096xf64>, index
    %6 = arith.addf %5, %4 : tensor<4096xf64>
    %c256 = arith.constant 256 : index
    %7 = tensor_ext.rotate %6, %c256 : tensor<4096xf64>, index
    %8 = arith.addf %7, %6 : tensor<4096xf64>
    %c128 = arith.constant 128 : index
    %9 = tensor_ext.rotate %8, %c128 : tensor<4096xf64>, index
    %10 = arith.addf %9, %8 : tensor<4096xf64>
    %11 = arith.addf %10, %arg2 : tensor<4096xf64>
    %12 = arith.mulf %11, %11 : tensor<4096xf64>
    %13 = tensor_ext.diagonal_matvec %12, %arg3 {diagonal_indices = array<i32: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127>, slots = 4096 : i64} : (tensor<4096xf64>, tensor<128x4096xf64>) -> tensor<4096xf64>
    %c2048_0 = arith.constant 2048 : index
    %14 = tensor_ext.rotate %13, %c2048_0 : tensor<4096xf64>, index
    %15 = arith.addf %14, %13 : tensor<4096xf64>
    %c1024_1 = arith.constant 1024 : index
    %16 = tensor_ext.rotate %15, %c1024_1 : tensor<4096xf64>, index
    %17 = arith.addf %16, %15 : tensor<4096xf64>
    %c512_2 = arith.constant 512 : index
    %18 = tensor_ext.rotate %17, %c512_2 : tensor<4096xf64>, index
    %19 = arith.addf %18, %17 : tensor<4096xf64>
    %c256_3 = arith.constant 256 : index
    %20 = tensor_ext.rotate %19, %c256_3 : tensor<4096xf64>, index
    %21 = arith.addf %20, %19 : tensor<4096xf64>
    %c128_4 = arith.constant 128 : index
    %22 = tensor_ext.rotate %21, %c128_4 : tensor<4096xf64>, index
    %23 = arith.addf %22, %21 : tensor<4096xf64>
    %24 = arith.addf %23, %arg4 : tensor<4096xf64>
    %25 = arith.mulf %24, %24 : tensor<4096xf64>
    %26 = tensor_ext.diagonal_matvec %25, %arg5 {diagonal_indices = array<i32: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 4087, 4088, 4089, 4090, 4091, 4092, 4093, 4094, 4095>, slots = 4096 : i64} : (tensor<4096xf64>, tensor<137x4096xf64>) -> tensor<4096xf64>
    %27 = arith.addf %26, %arg6 : tensor<4096xf64>
    return %27 : tensor<4096xf64>
  }
}
