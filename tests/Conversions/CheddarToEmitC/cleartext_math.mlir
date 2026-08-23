// RUN: heir-opt --cheddar-to-emitc %s | FileCheck %s
// RUN: heir-opt --cheddar-to-emitc %s | heir-translate --mlir-to-cpp | FileCheck --check-prefix=CPP %s

// CHECK: emitc.include <"cmath">
// CHECK: func.func @square_root
// CHECK: emitc.call_opaque "std::sqrt"
// CHECK: func.func @integer_max
// CHECK: emitc.cmp gt
// CHECK: emitc.conditional
// CHECK: func.func @float_max
// CHECK: emitc.cmp
// CHECK: emitc.conditional
// CHECK: func.func @fold_unit_extent
// CHECK-NOT: tensor.expand_shape
// CHECK: func.func @fold_unit_extent_slice
// CHECK-NOT: memref.collapse_shape

// CPP: #include <cmath>
// CPP: std::sqrt(

func.func @square_root(%arg0: f32) -> f32 {
  %0 = math.sqrt %arg0 : f32
  return %0 : f32
}

func.func @integer_max(%arg0: i32, %arg1: i32) -> i32 {
  %0 = arith.maxsi %arg0, %arg1 : i32
  return %0 : i32
}

func.func @float_max(%arg0: f32, %arg1: f32) -> f32 {
  %0 = arith.maximumf %arg0, %arg1 : f32
  return %0 : f32
}

func.func @fold_unit_extent(%arg0: tensor<100xf32>, %arg1: tensor<100xf32>)
    -> tensor<100xf32> {
  %0 = tensor.expand_shape %arg0 [[0, 1]] output_shape [1, 100]
      : tensor<100xf32> into tensor<1x100xf32>
  %1 = tensor.expand_shape %arg1 [[0, 1]] output_shape [1, 100]
      : tensor<100xf32> into tensor<1x100xf32>
  %2 = arith.subf %0, %1 : tensor<1x100xf32>
  %3 = tensor.collapse_shape %2 [[0, 1]]
      : tensor<1x100xf32> into tensor<100xf32>
  return %3 : tensor<100xf32>
}

func.func @fold_unit_extent_slice(%arg0: tensor<1x100xf32>,
    %arg1: tensor<1x100xf32>) -> tensor<100xf32> {
  %0 = arith.mulf %arg0, %arg1 : tensor<1x100xf32>
  %1 = tensor.extract_slice %0[0, 0] [1, 100] [1, 1]
      : tensor<1x100xf32> to tensor<100xf32>
  return %1 : tensor<100xf32>
}
