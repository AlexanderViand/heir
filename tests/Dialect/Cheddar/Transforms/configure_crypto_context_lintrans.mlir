// RUN: heir-opt --cheddar-configure-crypto-context=entry-function=main %s | FileCheck %s

// cheddar.linear_transform evaluates its BSGS rotations at the op's level, and
// CHEDDAR's level-specific key lookup (best-fit on the key-switch config) can
// reject a chain-max key for a much lower level. The configure function
// prepares each transform's rotations at the op's actual level. Rotations used
// only by linear transforms get NO chain-max copy (duplicates dominate key
// material on deep circuits); rotations any other op uses keep a chain-max key
// (hrot & co. look keys up without level constraints), and an LT at chain max
// sharing such a rotation adds nothing.

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
// The hrot's rotation keeps a chain-max key (Q has 6 primes -> maxLevel 5) ...
// CHECK: cheddar.prepare_rot_key
// CHECK-SAME: distance = 2
// CHECK-SAME: maxLevel = 5
// ... and the LT-only rotation is keyed per transform level (3 and 5), with
// no chain-max duplicate for the hrot-shared rotation's chain-max transform.
// CHECK: cheddar.prepare_rot_key
// CHECK-SAME: distance = 1
// CHECK-SAME: maxLevel = 3
// CHECK: cheddar.prepare_rot_key
// CHECK-SAME: distance = 1
// CHECK-SAME: maxLevel = 5
// CHECK-NOT: cheddar.prepare_rot_key
