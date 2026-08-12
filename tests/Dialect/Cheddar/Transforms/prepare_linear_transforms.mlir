// RUN: heir-opt --cheddar-prepare-linear-transforms --canonicalize %s | FileCheck %s

// CHECK: func.func @model__preprocessing(
// CHECK-SAME: %[[WEIGHTS:[a-zA-Z0-9_]+]]: tensor<2x4xf64>, %[[CTX:[a-zA-Z0-9_]+]]: !context)
// CHECK-SAME: -> (tensor<2x4xf64>, tensor<!linear_transform>)
// CHECK: %[[ONE:.*]] = arith.constant dense<1.000000e+00> : tensor<2x4xf64>
// CHECK: %[[DIAGS:.*]] = arith.addf %[[WEIGHTS]], %[[ONE]] : tensor<2x4xf64>
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<!linear_transform>
// CHECK: %[[PREPARED:.*]] = cheddar.prepare_linear_transform %[[CTX]], %[[DIAGS]], %[[EMPTY]]
// CHECK: return %[[WEIGHTS]], %[[PREPARED]]
func.func @model__preprocessing(%weights: tensor<2x4xf64>)
    -> tensor<2x4xf64> {
  return %weights : tensor<2x4xf64>
}

// CHECK: func.func @model__preprocessed(
// CHECK-SAME: %[[HANDLE:[a-zA-Z0-9_]+]]: tensor<!linear_transform>)
// CHECK-NOT: arith.addf
// CHECK-NOT: cheddar.linear_transform
// CHECK: cheddar.apply_prepared_linear_transform
// CHECK-SAME: %[[HANDLE]]
func.func @model__preprocessed(
    %ctx: !cheddar.context,
    %ct: tensor<!cheddar.ciphertext>,
    %evk: !cheddar.evk_map,
    %weights: tensor<2x4xf64>,
    %packed: tensor<2x4xf64>) -> tensor<!cheddar.ciphertext> {
  %one = arith.constant dense<1.0> : tensor<2x4xf64>
  %diags = arith.addf %weights, %one : tensor<2x4xf64>
  %out = tensor.empty() : tensor<!cheddar.ciphertext>
  %result = cheddar.linear_transform %ctx, %ct, %evk, %diags, %out
      {diagonal_indices = array<i32: 0, 1>, level = 5 : i64,
       bs = 2 : i64, gs = 1 : i64}
      : (!cheddar.context, tensor<!cheddar.ciphertext>, !cheddar.evk_map,
         tensor<2x4xf64>, tensor<!cheddar.ciphertext>)
      -> tensor<!cheddar.ciphertext>
  return %result : tensor<!cheddar.ciphertext>
}

// CHECK: func.func @model(
// CHECK: %[[PACK:.*]]:2 = call @model__preprocessing(%{{.*}}, %{{.*}})
// CHECK: call @model__preprocessed({{.*}}, %[[PACK]]#1)
func.func @model(
    %ctx: !cheddar.context,
    %ct: tensor<!cheddar.ciphertext>,
    %evk: !cheddar.evk_map,
    %weights: tensor<2x4xf64>) -> tensor<!cheddar.ciphertext> {
  %packed = call @model__preprocessing(%weights)
      : (tensor<2x4xf64>) -> tensor<2x4xf64>
  %result = call @model__preprocessed(%ctx, %ct, %evk, %weights, %packed)
      : (!cheddar.context, tensor<!cheddar.ciphertext>, !cheddar.evk_map,
         tensor<2x4xf64>, tensor<2x4xf64>)
      -> tensor<!cheddar.ciphertext>
  return %result : tensor<!cheddar.ciphertext>
}
