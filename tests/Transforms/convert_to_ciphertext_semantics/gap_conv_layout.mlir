// RUN: heir-opt %s --convert-to-ciphertext-semantics=ciphertext-size=2048 | FileCheck %s

// Gap-structured conv2d filter layout (stride-2 conv, Halevi-Shoup diagonals
// with floor-div gap structure). Regression test for the polyhedral blowups
// this layout used to trigger: Fourier-Motzkin in materializeLayout
// (std::bad_alloc after allocating ~100 GB) and parametric integer
// programming inside ISL for isDenseLayout / point enumeration (never
// terminated). The pass must complete quickly and materialize the
// 1024-ciphertext layout.

// A non-splat constant: the splat fast path (and its isDenseLayout check,
// which PIPs on this layout -- see google/heir#3193) must stay out of the
// way, exercising materializeLayout's bound computation and the constant
// packing enumeration.
// CHECK: func.func @gap_conv_filter
#layout = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : exists (e1, e2, e3, e4, e5, e6: i1 = 0 and 1024e5 = -i0 - 28i2 - i3 + ct + 784*floor((i0)/4) + 2e1 - 56e2 - 2e3 + 28e4 and 2048e6 = i0 + slot - 784*floor((i0)/4) - 2e1 - 28e4 and 0 <= i0 <= 7 and 0 <= i2 <= 1 and 0 <= i3 <= 1 and 0 <= ct <= 1023 and 0 <= slot <= 2047 and i0 <= 2e1 <= 27 + i0 and 0 <= e2 <= 13 and 0 <= e3 <= 13 and -1 - i0 + 2e1 <= 2e3 <= -i0 + 2e1 and 0 <= e4 <= 27 and -1 + i0 - 4*floor((i0)/4) + 4e2 <= 2e4 <= i0 - 4*floor((i0)/4) + 4e2) }">
module {
  func.func @gap_conv_filter() {
    %cst = arith.constant dense<[[[[1.0, 2.0], [3.0, 4.0]]], [[[5.0, 6.0], [7.0, 8.0]]], [[[1.5, 2.5], [3.5, 4.5]]], [[[5.5, 6.5], [7.5, 8.5]]], [[[1.25, 2.25], [3.25, 4.25]]], [[[5.25, 6.25], [7.25, 8.25]]], [[[1.75, 2.75], [3.75, 4.75]]], [[[5.75, 6.75], [7.75, 8.75]]]]> : tensor<8x1x2x2xf32>
    // CHECK: tensor<1024x2048xf32>
    %0 = secret.generic() {
      %1 = tensor_ext.assign_layout %cst {layout = #layout, tensor_ext.layout = #layout} : tensor<8x1x2x2xf32>
      secret.yield %1 : tensor<8x1x2x2xf32>
    } -> (!secret.secret<tensor<8x1x2x2xf32>> {tensor_ext.layout = #layout})
    return
  }
}
