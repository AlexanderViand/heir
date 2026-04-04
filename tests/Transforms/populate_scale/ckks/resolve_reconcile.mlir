// RUN: heir-opt %s --secret-insert-mgmt-ckks=before-mul-include-first-mul --resolve-reconcile-ckks | FileCheck %s --check-prefix=LOCAL
// RUN: heir-opt %s --secret-insert-mgmt-ckks=before-mul-include-first-mul --resolve-reconcile-ckks="reconcile-policy=canonical-per-level" | FileCheck %s --check-prefix=CANONICAL

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 14, Q = [36028797019389953, 35184372121601, 35184372744193], P = [36028797019488257, 36028797020209153], logDefaultScale = 45>, scheme.ckks} {
  func.func @mul(%arg0: !secret.secret<f32>) -> !secret.secret<f32> {
    %0 = secret.generic(%arg0 : !secret.secret<f32>) {
    ^body(%input0: f32):
      %1 = arith.mulf %input0, %input0 : f32
      %2 = arith.addf %1, %input0 : f32
      secret.yield %2 : f32
    } -> !secret.secret<f32>
    return %0 : !secret.secret<f32>
  }
}

// LOCAL: func.func @mul
// LOCAL: secret.generic
// LOCAL: ^body(%[[INPUT:.*]]: f32):
// LOCAL: %[[REDUCE_INPUT:.*]] = mgmt.modreduce %[[INPUT]]
// LOCAL-NEXT: %[[MUL:.*]] = arith.mulf %[[REDUCE_INPUT]], %[[REDUCE_INPUT]]
// LOCAL-NEXT: %[[RELIN:.*]] = mgmt.relinearize %[[MUL]]
// LOCAL-NEXT: %[[ADJUST:.*]] = mgmt.adjust_scale %[[INPUT]]
// LOCAL-NEXT: %[[REDUCE_ADJUST:.*]] = mgmt.modreduce %[[ADJUST]]
// LOCAL-NEXT: %[[ADD:.*]] = arith.addf %[[RELIN]], %[[REDUCE_ADJUST]]
// LOCAL-NEXT: %[[OUT:.*]] = mgmt.modreduce %[[ADD]]
// LOCAL-NEXT: secret.yield %[[OUT]]
// LOCAL-NOT: mgmt.reconcile
// LOCAL-NOT: mgmt.level_reduce

// CANONICAL: func.func @mul
// CANONICAL: secret.generic
// CANONICAL: ^body(%[[INPUT:.*]]: f32):
// CANONICAL: %[[REDUCE_INPUT:.*]] = mgmt.modreduce %[[INPUT]]
// CANONICAL-NEXT: %[[MUL:.*]] = arith.mulf %[[REDUCE_INPUT]], %[[REDUCE_INPUT]]
// CANONICAL-NEXT: %[[RELIN:.*]] = mgmt.relinearize %[[MUL]]
// CANONICAL-NEXT: %[[LEVEL_DOWN:.*]] = mgmt.level_reduce %[[INPUT]]
// CANONICAL-NEXT: %[[ADJUST:.*]] = mgmt.adjust_scale %[[LEVEL_DOWN]]
// CANONICAL-NEXT: %[[ADD:.*]] = arith.addf %[[RELIN]], %[[ADJUST]]
// CANONICAL-NEXT: %[[OUT:.*]] = mgmt.modreduce %[[ADD]]
// CANONICAL-NEXT: secret.yield %[[OUT]]
// CANONICAL-NOT: mgmt.reconcile
