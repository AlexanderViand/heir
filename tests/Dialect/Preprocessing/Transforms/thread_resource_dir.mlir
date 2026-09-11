// RUN: heir-opt --preprocessing-thread-resource-dir %s | FileCheck %s

// Loading functions and their callers gain a trailing directory argument; the
// loads read it and calls forward it.

// CHECK: func.func private @layout(%[[IN:.*]]: tensor<4xf32>, %[[DIR:.*]]: !preprocessing.resource_dir)
// CHECK: preprocessing.load_resource "weights.bin" from %[[DIR]] into %{{.*}} : (!preprocessing.resource_dir, tensor<4xf32>) -> tensor<4xf32>
func.func private @layout(%input: tensor<4xf32>) -> tensor<4xf32> {
  %empty = tensor.empty() : tensor<4xf32>
  %weights = preprocessing.load_resource "weights.bin" into %empty : (tensor<4xf32>) -> tensor<4xf32>
  %sum = arith.addf %input, %weights : tensor<4xf32>
  return %sum : tensor<4xf32>
}

// CHECK: func.func @preprocess(%[[IN:.*]]: tensor<4xf32>, %[[DIR:.*]]: !preprocessing.resource_dir)
// CHECK: call @layout(%[[IN]], %[[DIR]])
func.func @preprocess(%input: tensor<4xf32>) -> tensor<4xf32> {
  %result = call @layout(%input) : (tensor<4xf32>) -> tensor<4xf32>
  return %result : tensor<4xf32>
}

// A function that loads nothing is unchanged.
// CHECK: func.func @evaluate(%{{.*}}: tensor<4xf32>) -> tensor<4xf32>
func.func @evaluate(%input: tensor<4xf32>) -> tensor<4xf32> {
  return %input : tensor<4xf32>
}
