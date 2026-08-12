// RUN: heir-opt --cheddar-bufferize --fold-memref-alias-ops --cse --canonicalize %s | FileCheck %s

// Empty-tensor elimination propagates the caller-owned result destination
// through the rank-reducing insertion before One-Shot Bufferize. The encrypt
// operation therefore writes directly to the output element; neither the
// insertion nor materialization requires a payload copy.
// CHECK: func.func @packed_encrypt
// CHECK-SAME: %[[OUT:[a-zA-Z0-9_]+]]: memref<1x!ciphertext> {bufferize.result}
// CHECK-NOT: memref.alloc
// CHECK: %[[SLOT:[a-zA-Z0-9_]+]] = memref.subview %[[OUT]][0] [1] [1]
// CHECK: cheddar.encrypt %{{.*}}, %{{.*}}, %[[SLOT]]
// CHECK-NOT: memref.copy
// CHECK: return
func.func @packed_encrypt(
    %ui: !cheddar.user_interface,
    %plaintext: tensor<!cheddar.plaintext>)
    -> tensor<1x!cheddar.ciphertext> {
  %scalarInit = tensor.empty() : tensor<!cheddar.ciphertext>
  %encrypted = cheddar.encrypt %ui, %plaintext, %scalarInit
      : (!cheddar.user_interface, tensor<!cheddar.plaintext>,
         tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  %packedInit = tensor.empty() : tensor<1x!cheddar.ciphertext>
  %packed = tensor.insert_slice %encrypted into %packedInit[0] [1] [1]
      : tensor<!cheddar.ciphertext> into tensor<1x!cheddar.ciphertext>
  return %packed : tensor<1x!cheddar.ciphertext>
}

// A caller-provided aggregate result is threaded into the loop body. The
// scalar producer writes through a subview of it, so neither the loop boundary
// nor tensor.insert_slice needs a ciphertext copy.
// CHECK: func.func @loop_packed
// CHECK-SAME: %[[OUT:[a-zA-Z0-9_]+]]: memref<8x!ciphertext> {bufferize.result}
// CHECK-NOT: memref.alloc
// CHECK: scf.for
// CHECK: %[[SLOT:[a-zA-Z0-9_]+]] = memref.subview %[[OUT]][%{{.*}}] [1] [1]
// CHECK: cheddar.add %{{.*}}, %{{.*}}, %{{.*}}, %[[SLOT]]
// CHECK-NOT: cheddar.copy
// CHECK-NOT: memref.copy
// CHECK: return
func.func @loop_packed(%ctx: !cheddar.context,
                       %input: tensor<!cheddar.ciphertext>)
    -> tensor<8x!cheddar.ciphertext> {
  %output = tensor.empty() : tensor<8x!cheddar.ciphertext>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %result = scf.for %i = %c0 to %c8 step %c1
      iter_args(%iter = %output) -> tensor<8x!cheddar.ciphertext> {
    %empty = tensor.empty() : tensor<!cheddar.ciphertext>
    %value = cheddar.add %ctx, %input, %input, %empty
        : (!cheddar.context, tensor<!cheddar.ciphertext>,
           tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>)
            -> tensor<!cheddar.ciphertext>
    %inserted = tensor.insert_slice %value into %iter[%i] [1] [1]
        : tensor<!cheddar.ciphertext> into tensor<8x!cheddar.ciphertext>
    scf.yield %inserted : tensor<8x!cheddar.ciphertext>
  }
  return %result : tensor<8x!cheddar.ciphertext>
}
