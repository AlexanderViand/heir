// RUN: heir-opt --cheddar-configure-crypto-context=entry-function=main %s | FileCheck %s

// Bootstrap preparation must precede linear-transform key generation. The
// former has a large transient GPU-memory footprint, while the latter can
// leave hundreds of evaluation keys resident for a deep circuit.

!ciphertext = !cheddar.ciphertext
!boot_context = !cheddar.boot_context
!evk_map = !cheddar.evk_map
!ui = !cheddar.user_interface

module attributes {
  ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797018652673, 1125899907366913, 1125899907760129, 1125899908035841, 1125899908145153, 1125899908397057], P = [1152921504606994433], logDefaultScale = 45>,
  scheme.actual_slot_count = 4 : i64
} {
  func.func @main(%ctx: !boot_context, %ui: !ui, %ct: tensor<!ciphertext>,
                  %evk: !evk_map, %dg: tensor<2x4xf64>)
      -> tensor<!ciphertext> {
    %d0 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %booted = cheddar.boot %ctx, %ct, %evk, %d0
        : (!boot_context, tensor<!ciphertext>, !evk_map,
           tensor<!ciphertext>) -> tensor<!ciphertext>
    %d1 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %result = cheddar.linear_transform %ctx, %booted, %evk, %dg, %d1
        {diagonal_indices = array<i32: 0, 1>, level = 3 : i64,
         bs = 2 : i64, gs = 1 : i64}
        : (!boot_context, tensor<!ciphertext>, !evk_map, tensor<2x4xf64>,
           tensor<!ciphertext>) -> tensor<!ciphertext>
    return %result : tensor<!ciphertext>
  }
}

// CHECK: func.func @main__configure
// CHECK: cheddar.prepare_bootstrap
// CHECK: cheddar.prepare_rot_key
// CHECK-SAME: chainMaxLevel = 5
// CHECK-SAME: distance = 1
// CHECK-SAME: maxLevel = 3
