// RUN: heir-opt --layout-propagation --canonicalize %s | FileCheck %s

// Checks that a multi-ciphertext vector retains the explicit row-major
// conversion fallback before a matvec.
module {
  // CHECK: func.func @matvec
  func.func @matvec(%arg0: !secret.secret<tensor<14x14xf32>>, %arg1: !secret.secret<tensor<14x14xf32>>) -> !secret.secret<tensor<1x10xf32>> {
    %cst = arith.constant dense_resource<torch_tensor_10_torch.float32> : tensor<10xf32>
    %cst_0 = arith.constant dense_resource<torch_tensor_10_392_torch.float32> : tensor<10x392xf32>
    // CHECK: %[[cst:.*]] = arith.constant {{.*}} : tensor<1x2x14x14xf32>
    // CHECK: %[[v0:.*]] = tensor_ext.assign_layout %[[cst]]
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<1x2x14x14xf32>
    %cst_2 = arith.constant dense<4.000000e+00> : tensor<2x14x14xf32>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<10xf32>
    // CHECK: secret.generic
    // CHECK-NEXT: ^body(%[[input0:.*]]: tensor<14x14xf32>)
    %0 = secret.generic(%arg0: !secret.secret<tensor<14x14xf32>>) {
    ^body(%input0: tensor<14x14xf32>):
      // CHECK: %[[inserted_slice:.*]] = tensor.insert_slice %[[input0]] into %[[v0]]
      // CHECK: %[[inserted_slice_1:.*]] = tensor.insert_slice %[[input0]] into %[[inserted_slice]]
      // CHECK: %[[collapsed:.*]] = tensor.collapse_shape %[[inserted_slice_1]]
      // CHECK: %[[v4:.*]] = tensor_ext.convert_layout %[[collapsed]]
      // CHECK: linalg.matvec
      // CHECK-SAME: ins(%{{.*}}, %[[v4]] : tensor<10x392xf32>, tensor<392xf32>)
      %inserted_slice = tensor.insert_slice %input0 into %cst_1[0, 0, 0, 0] [1, 1, 14, 14] [1, 1, 1, 1] : tensor<14x14xf32> into tensor<1x2x14x14xf32>
      %inserted_slice_4 = tensor.insert_slice %input0 into %inserted_slice[0, 1, 0, 0] [1, 1, 14, 14] [1, 1, 1, 1] : tensor<14x14xf32> into tensor<1x2x14x14xf32>
      %collapsed = tensor.collapse_shape %inserted_slice_4 [[0, 1, 2, 3]] : tensor<1x2x14x14xf32> into tensor<392xf32>
      %2 = linalg.matvec ins(%cst_0, %collapsed : tensor<10x392xf32>, tensor<392xf32>) outs(%cst_3 : tensor<10xf32>) -> tensor<10xf32>
      %3 = arith.addf %2, %cst : tensor<10xf32>
      %expanded = tensor.expand_shape %3 [[0, 1]] output_shape [1, 10] : tensor<10xf32> into tensor<1x10xf32>
      secret.yield %expanded : tensor<1x10xf32>
    } -> !secret.secret<tensor<1x10xf32>>
    return %0 : !secret.secret<tensor<1x10xf32>>
  }
}

// A single-ciphertext, gap-packed convolution result is absorbed into the
// public matrix packing instead of being converted online before the matvec.
// CHECK: func.func @matvec_single_ct
func.func @matvec_single_ct(%arg0: !secret.secret<tensor<1x1x10x10xf32>>) -> !secret.secret<tensor<4xf32>> {
  %conv_init = arith.constant dense<0.0> : tensor<1x4x5x5xf32>
  %filter = arith.constant dense<0.25> : tensor<4x1x2x2xf32>
  %matrix = arith.constant dense<1.0> : tensor<4x100xf32>
  %bias = arith.constant dense<0.0> : tensor<4xf32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<1x1x10x10xf32>>) {
  ^body(%input0: tensor<1x1x10x10xf32>):
    // CHECK: secret.generic
    // CHECK: %[[conv:.*]] = linalg.conv_2d_nchw_fchw
    // CHECK: %[[collapsed:.*]] = tensor.collapse_shape %[[conv]]
    // CHECK-NOT: tensor_ext.convert_layout %[[collapsed]]
    // CHECK: linalg.matvec
    // CHECK-SAME: heir.matvec_input_slots = array<i64:
    // CHECK-SAME: ins(%{{.*}}, %[[collapsed]] : tensor<4x100xf32>, tensor<100xf32>)
    %conv = linalg.conv_2d_nchw_fchw {
      dilations = dense<1> : tensor<2xi64>,
      strides = dense<2> : tensor<2xi64>
    } ins(%input0, %filter : tensor<1x1x10x10xf32>, tensor<4x1x2x2xf32>)
      outs(%conv_init : tensor<1x4x5x5xf32>) -> tensor<1x4x5x5xf32>
    %collapsed = tensor.collapse_shape %conv [[0, 1, 2, 3]] :
      tensor<1x4x5x5xf32> into tensor<100xf32>
    %result = linalg.matvec
      ins(%matrix, %collapsed : tensor<4x100xf32>, tensor<100xf32>)
      outs(%bias : tensor<4xf32>) -> tensor<4xf32>
    secret.yield %result : tensor<4xf32>
  } -> !secret.secret<tensor<4xf32>>
  return %0 : !secret.secret<tensor<4xf32>>
}
