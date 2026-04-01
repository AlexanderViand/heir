// RUN: heir-translate --emit-cheddar %s | FileCheck %s

// CHECK: #include "core/Context.h"
// CHECK: using namespace cheddar;
// CHECK: using word = uint64_t;

// CHECK: Ct test_add
func.func @test_add(
    %ctx: !cheddar.context,
    %ct0: !cheddar.ciphertext,
    %ct1: !cheddar.ciphertext) -> !cheddar.ciphertext {
  // CHECK: Ct [[RES:.*]];
  // CHECK-NEXT: {{.*}}->Add([[RES]],
  %result = cheddar.add %ctx, %ct0, %ct1 : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// CHECK: Ct test_hmult
func.func @test_hmult(
    %ctx: !cheddar.context,
    %ct0: !cheddar.ciphertext,
    %ct1: !cheddar.ciphertext,
    %key: !cheddar.eval_key) -> !cheddar.ciphertext {
  // CHECK: Ct [[RES:.*]];
  // CHECK-NEXT: {{.*}}->HMult([[RES]], {{.*}}, {{.*}}, {{.*}}, true);
  %result = cheddar.hmult %ctx, %ct0, %ct1, %key {rescale = true} : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext, !cheddar.eval_key) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// CHECK: Ct test_hrot
func.func @test_hrot(
    %ctx: !cheddar.context,
    %ct: !cheddar.ciphertext,
    %key: !cheddar.eval_key) -> !cheddar.ciphertext {
  // CHECK: Ct [[RES:.*]];
  // CHECK-NEXT: {{.*}}->HRot([[RES]], {{.*}}, {{.*}}, 5);
  %result = cheddar.hrot %ctx, %ct, %key {static_shift = 5 : i64} : (!cheddar.context, !cheddar.ciphertext, !cheddar.eval_key) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}
