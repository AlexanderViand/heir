// RUN: heir-opt --mod-arith-to-arith-veir --reconcile-unrealized-casts %s | FileCheck %s

// REQUIRES: veir

// Lowers mod_arith to arith via the external (formally verified) veir-opt
// tool. The test is skipped unless veir-opt is available; see tests/lit.cfg.py
// for how the `veir` feature is detected.

!Zp = !mod_arith.int<65537 : i32>

// CHECK: @test_lower_add_mul
// CHECK-SAME: (%[[LHS:[^:]*]]: [[T:[^,]*]], %[[RHS:[^:]*]]: [[T]]) -> [[T]]
func.func @test_lower_add_mul(%lhs : !Zp, %rhs : !Zp) -> !Zp {
  // CHECK-NOT: mod_arith.add
  // CHECK-NOT: mod_arith.mul

  // The add is computed in a wider type to avoid overflow, then reduced.
  // CHECK-DAG: %[[CMOD_ADD:.*]] = arith.constant 65537 : i33
  // CHECK: %[[ADD:.*]] = arith.addi %{{.*}}, %{{.*}} : i33
  // CHECK: %[[ADD_REM:.*]] = arith.remui %[[ADD]], %[[CMOD_ADD]] : i33
  // CHECK: arith.trunci %[[ADD_REM]] overflow<nuw> : i33 to i32

  // The mul is computed in a doubled type to avoid overflow, then reduced.
  // CHECK-DAG: %[[CMOD_MUL:.*]] = arith.constant 65537 : i64
  // CHECK: %[[MUL:.*]] = arith.muli %{{.*}}, %{{.*}} : i64
  // CHECK: %[[MUL_REM:.*]] = arith.remui %[[MUL]], %[[CMOD_MUL]] : i64
  // CHECK: %[[RES:.*]] = arith.trunci %[[MUL_REM]] overflow<nuw> : i64 to i32

  // CHECK: %[[RES_CAST:.*]] = builtin.unrealized_conversion_cast %[[RES]] : i32 to [[T]]
  // CHECK: return %[[RES_CAST]] : [[T]]
  %add = mod_arith.add %lhs, %rhs : !Zp
  %mul = mod_arith.mul %add, %rhs : !Zp
  return %mul : !Zp
}
