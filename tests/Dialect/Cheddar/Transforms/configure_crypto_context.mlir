// RUN: heir-opt --cheddar-configure-crypto-context=entry-function=main %s | FileCheck %s

!ciphertext = !cheddar.ciphertext
!context = !cheddar.context
!ui = !cheddar.user_interface

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797018652673, 1125899907366913, 1125899907760129], P = [1152921504606994433], logDefaultScale = 45>} {
  func.func @main(%ctx: !context, %ui: !ui, %ct: tensor<!ciphertext>) -> tensor<!ciphertext> {
    %d0 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %rot = cheddar.hrot %ctx, %ui, %ct, %d0 {static_distance = 7 : i64} : (!context, !ui, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %d1 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %result = cheddar.hrot_add %ctx, %ui, %rot, %ct, %d1 {distance = 2 : i64} : (!context, !ui, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    return %result : tensor<!ciphertext>
  }
}

// CHECK: module attributes
// CHECK-SAME: cheddar.P = array<i64: 1152921504606994433>
// CHECK-SAME: cheddar.Q = array<i64: 36028797018652673, 1125899907366913, 1125899907760129>
// CHECK-SAME: cheddar.logDefaultScale = 45 : i64
// CHECK-SAME: cheddar.logN = 13 : i64
// CHECK-NOT: ckks.schemeParam
// CHECK: func.func @main__configure
// CHECK: cheddar.make_parameter
// CHECK: cheddar.create_context
// CHECK: cheddar.create_user_interface
// CHECK: cheddar.prepare_rot_key
// CHECK-SAME: distance = 2
// CHECK-SAME: maxLevel = 2
// CHECK: cheddar.prepare_rot_key
// CHECK-SAME: distance = 7
// CHECK-SAME: maxLevel = 2
