// RUN: heir-opt --layout-propagation %s | FileCheck %s

// Regression tests for two layout bugs around the gap-structured (pixel
// shuffled) result layout of strided convolutions:
//
// 1. The conv kernel adds its init (bias) operand directly to the kernel
//    output, which is packed per the gap layout — the init must be re-packed
//    into that layout instead of keeping its default row-major layout.
//
// 2. A downstream matvec consuming the gap-packed (collapsed) conv result
//    must absorb the slot permutation into its plaintext matrix layout
//    instead of converting the ciphertext (a shift network costing one
//    multiplicative level per stage).

// The composed gap result layout of the stride-2 conv (existential chain
// from the pixel-shuffle), assigned to the init below.
// CHECK-DAG: #[[init_layout:layout[0-9]*]] = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : exists (e1, e2, e3, e4:

// The matvec matrix layout with the vector's gap packing absorbed into its
// column space (rows of size 8 -> the "mod 8" replication constraint).
// CHECK-DAG: #[[mat_layout:layout[0-9]*]] = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : exists (e0, e1, e2: (-i0 + slot) mod 8 = 0

// CHECK: @conv2d_gap_init
func.func @conv2d_gap_init(%arg0: !secret.secret<tensor<1x1x10x10xf32>>) -> !secret.secret<tensor<1x4x5x5xf32>> {
  %filter = arith.constant dense<2.500000e-01> : tensor<4x1x2x2xf32>
  %bias = arith.constant dense<1.000000e+00> : tensor<1x4x5x5xf32>

  %0 = secret.generic(%arg0 : !secret.secret<tensor<1x1x10x10xf32>>) {
  ^body(%input0: tensor<1x1x10x10xf32>):
    // The bias (init) is re-packed into the conv kernel's gap result layout
    // rather than keeping its default row-major layout.
    // CHECK: %[[init:[^ ]+]] = tensor_ext.assign_layout
    // CHECK-SAME: layout = #[[init_layout]]
    // CHECK: linalg.conv_2d_nchw_fchw
    // CHECK-SAME: outs(%[[init]]
    %1 = linalg.conv_2d_nchw_fchw
      { dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64> }
      ins(%input0, %filter : tensor<1x1x10x10xf32>, tensor<4x1x2x2xf32>)
      outs(%bias : tensor<1x4x5x5xf32>) -> tensor<1x4x5x5xf32>
    secret.yield %1 : tensor<1x4x5x5xf32>
  } -> !secret.secret<tensor<1x4x5x5xf32>>
  return %0 : !secret.secret<tensor<1x4x5x5xf32>>
}

// -----

// CHECK: @conv2d_gap_matvec
func.func @conv2d_gap_matvec(%arg0: !secret.secret<tensor<1x1x32x32xf32>>) -> !secret.secret<tensor<8xf32>> {
  %filter = arith.constant dense<2.500000e-01> : tensor<4x1x2x2xf32>
  %conv_init = arith.constant dense<0.000000e+00> : tensor<1x4x16x16xf32>
  %weights = arith.constant dense<1.000000e-01> : tensor<8x1024xf32>
  %mv_init = arith.constant dense<0.000000e+00> : tensor<8xf32>

  %0 = secret.generic(%arg0 : !secret.secret<tensor<1x1x32x32xf32>>) {
  ^body(%input0: tensor<1x1x32x32xf32>):
    %1 = linalg.conv_2d_nchw_fchw
      { dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64> }
      ins(%input0, %filter : tensor<1x1x32x32xf32>, tensor<4x1x2x2xf32>)
      outs(%conv_init : tensor<1x4x16x16xf32>) -> tensor<1x4x16x16xf32>
    // The collapsed gap-packed conv result feeds the matvec directly: the
    // slot permutation is absorbed into the plaintext matrix layout and no
    // ciphertext-side layout conversion is inserted.
    // CHECK: %[[collapsed:[^ ]+]] = tensor.collapse_shape
    // CHECK-NOT: tensor_ext.convert_layout
    // CHECK: tensor_ext.assign_layout
    // CHECK-SAME: layout = #[[mat_layout]]
    // CHECK: linalg.matvec
    // CHECK-SAME: %[[collapsed]]
    %2 = tensor.collapse_shape %1 [[0, 1, 2, 3]] : tensor<1x4x16x16xf32> into tensor<1024xf32>
    %3 = linalg.matvec ins(%weights, %2 : tensor<8x1024xf32>, tensor<1024xf32>) outs(%mv_init : tensor<8xf32>) -> tensor<8xf32>
    secret.yield %3 : tensor<8xf32>
  } -> !secret.secret<tensor<8xf32>>
  return %0 : !secret.secret<tensor<8xf32>>
}
