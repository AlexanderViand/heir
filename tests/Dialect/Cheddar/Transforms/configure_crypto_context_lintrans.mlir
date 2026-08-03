// RUN: heir-opt --cheddar-configure-crypto-context=entry-function=main %s | FileCheck %s

// cheddar.linear_transform evaluates its BSGS rotations at the op's level, and
// CHEDDAR's level-specific key lookup (best-fit on the key-switch config) can
// reject a chain-max key for a much lower level. The configure function must
// therefore prepare each transform's rotations at the op's actual level, in
// addition to the chain-max keys from RotationAnalysis. A transform already at
// chain-max must not add duplicate keys.

!ciphertext = !cheddar.ciphertext
!context = !cheddar.context
!evk_map = !cheddar.evk_map

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797018652673, 1125899907366913, 1125899907760129, 1125899908035841, 1125899908145153, 1125899908397057], P = [1152921504606994433], logDefaultScale = 45>} {
  func.func @main(%ctx: !context, %ct: tensor<!ciphertext>, %evk: !evk_map, %dg: tensor<2x4xf64>) -> tensor<!ciphertext> {
    %d0 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %r = cheddar.linear_transform %ctx, %ct, %evk, %dg, %d0 {diagonal_indices = array<i32: 0, 1>, level = 3 : i64, bs = 2 : i64, gs = 1 : i64} : (!context, tensor<!ciphertext>, !evk_map, tensor<2x4xf64>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %d1 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %s = cheddar.linear_transform %ctx, %r, %evk, %dg, %d1 {diagonal_indices = array<i32: 0, 1>, level = 5 : i64, bs = 2 : i64, gs = 1 : i64} : (!context, tensor<!ciphertext>, !evk_map, tensor<2x4xf64>, tensor<!ciphertext>) -> tensor<!ciphertext>
    return %s : tensor<!ciphertext>
  }
}

// CHECK: func.func @main__configure
// The program rotations at chain max (Q has 6 primes -> maxLevel 5) ...
// CHECK: cheddar.prepare_rot_key
// CHECK-SAME: distance = 1
// CHECK-SAME: maxLevel = 5
// ... plus the level-3 transform's rotations at its own level. The level-5
// transform is already covered by the chain-max key and adds nothing.
// CHECK: cheddar.prepare_rot_key
// CHECK-SAME: distance = 1
// CHECK-SAME: maxLevel = 3
// CHECK-NOT: cheddar.prepare_rot_key
