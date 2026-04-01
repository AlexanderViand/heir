// RUN: heir-opt %s | FileCheck %s

// Test that the cheddar dialect can be parsed and printed.

// CHECK: @test_add
func.func @test_add(
    %ctx: !cheddar.context,
    %ct0: !cheddar.ciphertext,
    %ct1: !cheddar.ciphertext) -> !cheddar.ciphertext {
  // CHECK: cheddar.add
  %result = cheddar.add %ctx, %ct0, %ct1 : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// CHECK: @test_hmult
func.func @test_hmult(
    %ctx: !cheddar.context,
    %ct0: !cheddar.ciphertext,
    %ct1: !cheddar.ciphertext,
    %key: !cheddar.eval_key) -> !cheddar.ciphertext {
  // CHECK: cheddar.hmult
  %result = cheddar.hmult %ctx, %ct0, %ct1, %key {rescale = true} : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext, !cheddar.eval_key) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// CHECK: @test_hrot
func.func @test_hrot(
    %ctx: !cheddar.context,
    %ct: !cheddar.ciphertext,
    %key: !cheddar.eval_key) -> !cheddar.ciphertext {
  // CHECK: cheddar.hrot
  %result = cheddar.hrot %ctx, %ct, %key {static_shift = 5 : i64} : (!cheddar.context, !cheddar.ciphertext, !cheddar.eval_key) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// CHECK: @test_rescale
func.func @test_rescale(
    %ctx: !cheddar.context,
    %ct: !cheddar.ciphertext) -> !cheddar.ciphertext {
  // CHECK: cheddar.rescale
  %result = cheddar.rescale %ctx, %ct : (!cheddar.context, !cheddar.ciphertext) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}

// CHECK: @test_boot
func.func @test_boot(
    %ctx: !cheddar.context,
    %ct: !cheddar.ciphertext,
    %evk: !cheddar.evk_map) -> !cheddar.ciphertext {
  // CHECK: cheddar.boot
  %result = cheddar.boot %ctx, %ct, %evk : (!cheddar.context, !cheddar.ciphertext, !cheddar.evk_map) -> !cheddar.ciphertext
  return %result : !cheddar.ciphertext
}
