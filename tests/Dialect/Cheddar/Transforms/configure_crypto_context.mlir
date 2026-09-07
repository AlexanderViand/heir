// RUN: heir-opt --cheddar-configure-crypto-context=entry-function=main %s | FileCheck %s
// RUN: heir-opt --cheddar-configure-crypto-context="entry-function=main prepare-rotation-keys-at-use-levels=true" %s | FileCheck %s --check-prefix=USE-LEVELS

!ciphertext = !cheddar.ciphertext
!context = !cheddar.context
!ui = !cheddar.user_interface
!evk_map = !cheddar.evk_map

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797018652673, 1125899907366913, 1125899907760129], P = [1152921504606994433], logDefaultScale = 45>} {
  func.func @main(%ctx: !context, %ui: !ui, %ct: tensor<!ciphertext>, %evk: !evk_map) -> tensor<!ciphertext> {
    %d0 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %rot = cheddar.hrot %ctx, %evk, %ct, %d0 {level = 1 : i64, static_distance = 7 : i64} : (!context, !evk_map, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %d1 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %result = cheddar.hrot_add %ctx, %evk, %rot, %ct, %d1 {distance = 2 : i64, level = 0 : i64} : (!context, !evk_map, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %d2 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %result2 = cheddar.hrot_add %ctx, %evk, %result, %ct, %d2 {distance = 7 : i64, level = 0 : i64} : (!context, !evk_map, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    return %result2 : tensor<!ciphertext>
  }
}

// CHECK: module attributes
// CHECK-SAME: cheddar.P = array<i64: 1152921504606994433>
// CHECK-SAME: cheddar.Q = array<i64: 36028797018652673, 1125899907366913, 1125899907760129>
// CHECK-SAME: cheddar.logDefaultScale = 45 : i64
// CHECK-SAME: cheddar.logN = 13 : i64
// CHECK-NOT: ckks.schemeParam
// CHECK: func.func @main__setup
// CHECK-SAME: client.setup_func = {func_name = "main"}
// CHECK: cheddar.make_parameter
// CHECK: cheddar.create_context
// CHECK: func.func @main__keygen
// CHECK-SAME: client.keygen_func = {func_name = "main"}
// CHECK: cheddar.create_user_interface
// CHECK: cheddar.prepare_rot_key
// CHECK-SAME: distance = 2
// CHECK-SAME: maxLevel = 2
// CHECK: cheddar.prepare_rot_key
// CHECK-SAME: distance = 7
// CHECK-SAME: maxLevel = 2
// CHECK-NOT: cheddar.prepare_rot_key
// CHECK: return %{{.*}}, %{{.*}} : tensor<!context>, tensor<!user_interface>
// CHECK: func.func @main__configure
// CHECK: call @main__setup
// CHECK: call @main__keygen

// USE-LEVELS: func.func @main__keygen
// USE-LEVELS: cheddar.prepare_rot_key
// USE-LEVELS-SAME: distance = 2
// USE-LEVELS-SAME: maxLevel = 0
// USE-LEVELS: cheddar.prepare_rot_key
// USE-LEVELS-SAME: distance = 7
// USE-LEVELS-SAME: maxLevel = 1
// USE-LEVELS: cheddar.prepare_rot_key
// USE-LEVELS-SAME: distance = 7
// USE-LEVELS-SAME: maxLevel = 0
