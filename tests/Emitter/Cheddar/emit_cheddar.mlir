// RUN: heir-translate --emit-cheddar %s | FileCheck %s

// CHECK: #include <cheddar/include/core/Context.h>
// CHECK: using namespace cheddar;
// CHECK: using word = uint64_t;

// CHECK: Ct test_add
func.func @test_add(
    %ctx: !cheddar.context,
    %ct0: !cheddar.ciphertext,
    %ct1: !cheddar.ciphertext) -> !cheddar.ciphertext {
  // CHECK: Ct [[RES:v[0-9]+]];
  // CHECK-NEXT: [[CTX:v[0-9]+]]->Add([[RES]], [[LHS:v[0-9]+]], [[RHS:v[0-9]+]]);
  %result = cheddar.add %ctx, %ct0, %ct1 : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// CHECK: Ct test_hmult
func.func @test_hmult(
    %ctx: !cheddar.context,
    %ct0: !cheddar.ciphertext,
    %ct1: !cheddar.ciphertext,
    %key: !cheddar.eval_key) -> !cheddar.ciphertext {
  // CHECK: Ct [[RES:v[0-9]+]];
  // CHECK-NEXT: [[CTX:v[0-9]+]]->HMult([[RES]], {{.*}}, {{.*}}, {{.*}}, true);
  %result = cheddar.hmult %ctx, %ct0, %ct1, %key {rescale = true} : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext, !cheddar.eval_key) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// CHECK: Ct test_hrot
func.func @test_hrot(
    %ctx: !cheddar.context,
    %ct: !cheddar.ciphertext,
    %key: !cheddar.eval_key) -> !cheddar.ciphertext {
  // CHECK: Ct [[RES:v[0-9]+]];
  // CHECK-NEXT: [[CTX:v[0-9]+]]->HRot([[RES]], {{.*}}, {{.*}}, 5);
  %result = cheddar.hrot %ctx, %ct, %key {distance = 5 : i64} : (!cheddar.context, !cheddar.ciphertext, !cheddar.eval_key) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}
