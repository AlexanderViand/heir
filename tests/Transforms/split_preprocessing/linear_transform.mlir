// RUN: heir-opt --split-preprocessing %s | FileCheck %s

// CHECK: func.func @dynamic_linear_transform__preprocessing(%{{.*}}: tensor<2x4xf32>)
// CHECK-SAME: server.preprocessing_func = {entry_arg_indices = array<i64: 1>, func_name = "dynamic_linear_transform"}
// CHECK: func.func @dynamic_linear_transform__preprocessed(
// CHECK-SAME: %[[DIAGS:.*]]: tensor<2x4xf32>
// CHECK: kernel.linear_transform %{{.*}}, %[[DIAGS:[a-zA-Z0-9_]+]]
// CHECK: func.func @dynamic_linear_transform(
// CHECK: call @dynamic_linear_transform__preprocessing(%{{.*}})
// CHECK: call @dynamic_linear_transform__preprocessed
func.func @dynamic_linear_transform(%input: tensor<4xf32>,
                                    %diagonals: tensor<2x4xf32>)
    -> tensor<4xf32> {
  %result = kernel.linear_transform %input, %diagonals
      {diagonal_indices = array<i64: 0, 1>}
      : tensor<4xf32>, tensor<2x4xf32> -> tensor<4xf32>
  return %result : tensor<4xf32>
}
