// RUN: heir-opt --annotate-module="backend=cheddar scheme=ckks" --mlir-to-ckks="min-slot-count=8192 greedy-level-budget=6 greedy-bootstrap-waterline=3" --scheme-to-cheddar="entry-function=bootstrap log-message-ratio=1" %s | FileCheck %s

// Exercise parameter generation and context configuration together. The
// generated Q chain must be deep enough for 4 CtS + 8 EvalMod + 2 StC levels;
// otherwise ConfigureCryptoContext rejects this pipeline before producing the
// setup operations below.

// CHECK: module attributes {
// CHECK-SAME: backend.cheddar
// CHECK-SAME: cheddar.boot.num_cts = 4 : i64
// CHECK-SAME: cheddar.boot.num_stc = 2 : i64
// CHECK: func.func @bootstrap(
// CHECK: cheddar.boot
// CHECK: func.func @bootstrap__setup
// CHECK: cheddar.make_parameter
// CHECK-SAME: defaultEncryptionLevel =
// CHECK: cheddar.create_boot_context
// CHECK-SAME: logMessageRatio = 1
// CHECK-SAME: numCtsLevels = 4
// CHECK-SAME: numStcLevels = 2
// CHECK: func.func @bootstrap__keygen
// CHECK: cheddar.prepare_bootstrap
// CHECK: func.func @bootstrap__configure
// CHECK: call @bootstrap__setup
// CHECK: call @bootstrap__keygen
func.func @bootstrap(%input: tensor<1024xf32> {secret.secret})
    -> tensor<1024xf32> {
  %0 = arith.mulf %input, %input : tensor<1024xf32>
  %1 = arith.mulf %0, %0 : tensor<1024xf32>
  %2 = arith.mulf %1, %1 : tensor<1024xf32>
  %3 = arith.mulf %2, %2 : tensor<1024xf32>
  %4 = arith.mulf %3, %3 : tensor<1024xf32>
  %5 = arith.mulf %4, %4 : tensor<1024xf32>
  %result = arith.mulf %5, %input : tensor<1024xf32>
  return %result : tensor<1024xf32>
}
