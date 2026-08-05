// RUN: heir-opt --cheddar-configure-crypto-context=entry-function=main %s | FileCheck %s

// CHEDDAR holds one rotation key per index, looked up with no level
// constraint, so the configure function prepares each distinct rotation index
// exactly once at chain max -- linear-transform rotations included (transforms
// evaluate with min_ks, which is correct with chain-max keys at any level).

!ciphertext = !cheddar.ciphertext
!context = !cheddar.context
!evk_map = !cheddar.evk_map
!ui = !cheddar.user_interface

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797018652673, 1125899907366913, 1125899907760129, 1125899908035841, 1125899908145153, 1125899908397057], P = [1152921504606994433], logDefaultScale = 45>} {
  func.func @main(%ctx: !context, %ui: !ui, %ct: tensor<!ciphertext>, %evk: !evk_map, %dg: tensor<2x4xf64>, %dg3: tensor<2x4xf64>) -> tensor<!ciphertext> {
    // Rotation 1 is LT-only (levels 3 and 5); rotation 2 is used by an hrot.
    %d0 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %r = cheddar.linear_transform %ctx, %ct, %evk, %dg, %d0 {diagonal_indices = array<i32: 0, 1>, level = 3 : i64, bs = 2 : i64, gs = 1 : i64} : (!context, tensor<!ciphertext>, !evk_map, tensor<2x4xf64>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %d1 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %s = cheddar.linear_transform %ctx, %r, %evk, %dg, %d1 {diagonal_indices = array<i32: 0, 1>, level = 5 : i64, bs = 2 : i64, gs = 1 : i64} : (!context, tensor<!ciphertext>, !evk_map, tensor<2x4xf64>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %d2 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %t = cheddar.hrot %ctx, %ui, %s, %d2 {static_distance = 2 : i64} : (!context, !ui, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    // An LT at chain max whose rotation the hrot also uses: covered by the
    // chain-max key, adds nothing.
    %d3 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %u = cheddar.linear_transform %ctx, %t, %evk, %dg3, %d3 {diagonal_indices = array<i32: 0, 2>, level = 5 : i64, bs = 3 : i64, gs = 1 : i64} : (!context, tensor<!ciphertext>, !evk_map, tensor<2x4xf64>, tensor<!ciphertext>) -> tensor<!ciphertext>
    return %u : tensor<!ciphertext>
  }
}

// CHECK: func.func @main__configure
// One key per distinct rotation index (Q has 6 primes -> maxLevel 5): the
// hrot's rotation 2 and the transforms' rotation 1, each exactly once.
// CHECK-DAG: cheddar.prepare_rot_key {{.*}}distance = 2 : i64, maxLevel = 5
// CHECK-DAG: cheddar.prepare_rot_key {{.*}}distance = 1 : i64, maxLevel = 5
// CHECK-NOT: cheddar.prepare_rot_key
