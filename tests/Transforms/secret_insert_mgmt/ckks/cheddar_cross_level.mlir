// RUN: heir-opt --secret-insert-mgmt-ckks=slot-number=8 %s | FileCheck %s

// In cheddar mode (backend.cheddar) a cross-level addition must be resolved by
// bringing the higher-level operand down with a level_reduce (lowered later to
// cheddar.level_down), NOT by an adjust_scale. adjust_scale lowers to a scalar
// mul_plain with the nominal scale, which Cheddar rejects. Cheddar's fixed
// canonical scale per level makes the plain level reduction scale-correct.
//
// Two chained multiplications create a genuine level gap: with rescale-after-
// mult (forced in cheddar mode) %m2 sits a level below %m1, so adding them is a
// cross-level add that must be resolved with level_reduce, not adjust_scale.

// CHECK: func @cross_level
// CHECK-NOT: mgmt.adjust_scale
// CHECK: mgmt.level_reduce

module attributes {backend.cheddar, scheme.ckks} {
  func.func @cross_level(%arg0: !secret.secret<tensor<8xf32>>, %arg1: !secret.secret<tensor<8xf32>>) -> !secret.secret<tensor<8xf32>> {
    %0 = secret.generic(%arg0 : !secret.secret<tensor<8xf32>>, %arg1 : !secret.secret<tensor<8xf32>>) {
    ^body(%input0: tensor<8xf32>, %input1: tensor<8xf32>):
      %m1 = arith.mulf %input0, %input1 : tensor<8xf32>
      %m2 = arith.mulf %m1, %m1 : tensor<8xf32>
      // cross-level add: %m1 is one level above %m2.
      %r = arith.addf %m2, %m1 : tensor<8xf32>
      secret.yield %r : tensor<8xf32>
    } -> !secret.secret<tensor<8xf32>>
    return %0 : !secret.secret<tensor<8xf32>>
  }
}
