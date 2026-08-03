// A stride-2 NCHW conv2d in miniature, mirroring the LoLA torch export
// exactly (broadcast bias as the conv init, stride 2, no padding, output
// channels padded to stride^2 with a zero filter channel). Value-asserting
// regression test for the gap-packed conv2d kernel.
module {
  func.func @conv2d_stride2(%arg0: tensor<1x1x6x6xf32> {secret.secret}) -> tensor<1x4x3x3xf32> {
    %filter = arith.constant dense<[[[[1.000000e+00, -1.000000e+00], [5.000000e-01, 2.500000e-01]]], [[[-5.000000e-01, 1.000000e+00], [1.000000e+00, -1.000000e+00]]], [[[2.500000e-01, 5.000000e-01], [-2.500000e-01, 7.500000e-01]]], [[[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]]]]> : tensor<4x1x2x2xf32>
    %bias = arith.constant dense<[1.000000e-01, -2.000000e-01, 3.000000e-01, 0.000000e+00]> : tensor<4xf32>
    %0 = tensor.empty() : tensor<1x4x3x3xf32>
    %broadcasted = linalg.broadcast ins(%bias : tensor<4xf32>) outs(%0 : tensor<1x4x3x3xf32>) dimensions = [0, 2, 3]
    %1 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%arg0, %filter : tensor<1x1x6x6xf32>, tensor<4x1x2x2xf32>) outs(%broadcasted : tensor<1x4x3x3xf32>) -> tensor<1x4x3x3xf32>
    return %1 : tensor<1x4x3x3xf32>
  }
}
