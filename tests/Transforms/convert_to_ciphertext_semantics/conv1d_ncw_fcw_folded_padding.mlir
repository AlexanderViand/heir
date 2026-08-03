// RUN: heir-opt --layout-propagation=ciphertext-size=1024 %s | FileCheck %s --check-prefix=PROP
// RUN: heir-opt --layout-propagation=ciphertext-size=1024 --convert-to-ciphertext-semantics=ciphertext-size=1024 %s | FileCheck %s --check-prefix=CTS

// The fold must be recorded on the op, because the matrix is then 32x48 (8
// output channels * 4 output positions, by 8 input channels * the *unpadded*
// width 6) rather than the 32x80 that the op's own padded operand implies. Both
// round to 32 rows, but to 64 vs 128 columns, and the squat-diagonal collapse
// runs log2(cols) - log2(rows) rotate-and-add rounds starting at cols/2.
// Deriving 128 columns in the lowering yields two rounds starting at 64 instead
// of one at 32, i.e. silently wrong output.

// PROP-NOT: tensor_ext.convert_layout
// PROP: linalg.conv_1d_ncw_fcw
// PROP-SAME: heir.conv_folded_padding = 2 : i64

// The collapse: exactly one rotate-and-add, by 64/2 = 32. A rotation by 64
// anywhere would mean the padded column count leaked back in.
// CTS-NOT: %c64
// CTS: %[[collapse:.*]] = tensor_ext.rotate %[[sum:.*]], %c32
// CTS-NEXT: arith.addf %[[sum]], %[[collapse]]

func.func @conv1d_padded_stride2(%arg0: !secret.secret<tensor<1x8x6xf32>>) -> !secret.secret<tensor<1x8x4xf32>> {
  %out = arith.constant dense<0.000000e+00> : tensor<1x8x4xf32>
  %filter = arith.constant dense<2.000000e+00> : tensor<8x8x3xf32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<1x8x6xf32>>) {
  ^body(%input0: tensor<1x8x6xf32>):
    %zero = arith.constant 0.000000e+00 : f32
    %padded = tensor.pad %input0 low[0, 0, 2] high[0, 0, 2] {
    ^bb0(%i: index, %j: index, %k: index):
      tensor.yield %zero : f32
    } : tensor<1x8x6xf32> to tensor<1x8x10xf32>
    %1 = linalg.conv_1d_ncw_fcw {dilations = dense<1> : vector<1xi64>, strides = dense<2> : vector<1xi64>} ins(%padded, %filter : tensor<1x8x10xf32>, tensor<8x8x3xf32>) outs(%out : tensor<1x8x4xf32>) -> tensor<1x8x4xf32>
    secret.yield %1 : tensor<1x8x4xf32>
  } -> !secret.secret<tensor<1x8x4xf32>>
  return %0 : !secret.secret<tensor<1x8x4xf32>>
}
