// RUN: heir-opt --cheddar-prepare-linear-transforms --canonicalize %s | FileCheck %s

// CHECK: func.func @model__preprocessing(
// CHECK-SAME: %[[DIAGS:.*]]: tensor<2x4xf64>, %[[CTX:.*]]: !context)
// CHECK: cheddar.prepare_linear_transform %[[CTX]], %[[DIAGS]]
func.func @model__preprocessing(%diagonals: tensor<2x4xf64>)
    -> tensor<2x4xf64> {
  return %diagonals : tensor<2x4xf64>
}

// CHECK: func.func @model__preprocessed(
// CHECK-SAME: %[[HANDLE:[a-zA-Z0-9_]+]]: tensor<!linear_transform>)
// CHECK: cheddar.apply_prepared_linear_transform
// CHECK-SAME: %[[HANDLE]]
func.func @model__preprocessed(
    %ctx: !cheddar.context, %ct: tensor<!cheddar.ciphertext>,
    %evk: !cheddar.evk_map, %diagonals: tensor<2x4xf64>,
    %packed: tensor<2x4xf64>) -> tensor<!cheddar.ciphertext> {
  %out = tensor.empty() : tensor<!cheddar.ciphertext>
  %result = cheddar.linear_transform %ctx, %ct, %evk, %diagonals, %out
      {diagonal_indices = array<i32: 0, 1>, level = 5 : i64,
       bs = 2 : i64, gs = 1 : i64}
      : (!cheddar.context, tensor<!cheddar.ciphertext>, !cheddar.evk_map,
         tensor<2x4xf64>, tensor<!cheddar.ciphertext>)
      -> tensor<!cheddar.ciphertext>
  return %result : tensor<!cheddar.ciphertext>
}

// CHECK: func.func @model(
// CHECK: call @model__preprocessing
// CHECK: call @model__preprocessed
func.func @model(
    %ctx: !cheddar.context, %ct: tensor<!cheddar.ciphertext>,
    %evk: !cheddar.evk_map,
    %diagonals: tensor<2x4xf64>) -> tensor<!cheddar.ciphertext> {
  %packed = call @model__preprocessing(%diagonals)
      : (tensor<2x4xf64>) -> tensor<2x4xf64>
  %result = call @model__preprocessed(%ctx, %ct, %evk, %diagonals, %packed)
      : (!cheddar.context, tensor<!cheddar.ciphertext>, !cheddar.evk_map,
         tensor<2x4xf64>, tensor<2x4xf64>)
      -> tensor<!cheddar.ciphertext>
  return %result : tensor<!cheddar.ciphertext>
}
