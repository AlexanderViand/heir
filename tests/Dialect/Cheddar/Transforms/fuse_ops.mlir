// RUN: heir-opt --cheddar-fuse-ops %s | FileCheck %s

// mult + relinearize + rescale -> hmult (rescale=true, elided as the default)
// CHECK: @fuse_hmult_rescale
func.func @fuse_hmult_rescale(
    %ctx: !cheddar.context, %ct0: !cheddar.ciphertext,
    %ct1: !cheddar.ciphertext, %key: !cheddar.eval_key) -> !cheddar.ciphertext {
  // CHECK-NOT: cheddar.mult
  // CHECK-NOT: cheddar.relinearize
  // CHECK-NOT: cheddar.rescale
  // CHECK: cheddar.hmult
  // CHECK-NOT: rescale = false
  %mult = cheddar.mult %ctx, %ct0, %ct1 : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
  %relin = cheddar.relinearize %ctx, %mult, %key : (!cheddar.context, !cheddar.ciphertext, !cheddar.eval_key) -> !cheddar.ciphertext
  %rescaled = cheddar.rescale %ctx, %relin : (!cheddar.context, !cheddar.ciphertext) -> !cheddar.ciphertext
  return %rescaled : !cheddar.ciphertext
}

// mult + relinearize -> hmult (rescale=false)
// CHECK: @fuse_hmult_no_rescale
func.func @fuse_hmult_no_rescale(
    %ctx: !cheddar.context, %ct0: !cheddar.ciphertext,
    %ct1: !cheddar.ciphertext, %key: !cheddar.eval_key) -> !cheddar.ciphertext {
  // CHECK-NOT: cheddar.mult
  // CHECK-NOT: cheddar.relinearize
  // CHECK: cheddar.hmult
  // CHECK-SAME: rescale = false
  %mult = cheddar.mult %ctx, %ct0, %ct1 : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
  %relin = cheddar.relinearize %ctx, %mult, %key : (!cheddar.context, !cheddar.ciphertext, !cheddar.eval_key) -> !cheddar.ciphertext
  return %relin : !cheddar.ciphertext
}

// mult + relinearize_rescale -> hmult (rescale=true)
// CHECK: @fuse_hmult_relin_rescale
func.func @fuse_hmult_relin_rescale(
    %ctx: !cheddar.context, %ct0: !cheddar.ciphertext,
    %ct1: !cheddar.ciphertext, %key: !cheddar.eval_key) -> !cheddar.ciphertext {
  // CHECK-NOT: cheddar.mult
  // CHECK-NOT: cheddar.relinearize_rescale
  // CHECK: cheddar.hmult
  %mult = cheddar.mult %ctx, %ct0, %ct1 : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
  %rr = cheddar.relinearize_rescale %ctx, %mult, %key : (!cheddar.context, !cheddar.ciphertext, !cheddar.eval_key) -> !cheddar.ciphertext
  return %rr : !cheddar.ciphertext
}

// hrot (static distance) + add -> hrot_add
// CHECK: @fuse_hrot_add_static
func.func @fuse_hrot_add_static(
    %ctx: !cheddar.context, %ct0: !cheddar.ciphertext,
    %ct1: !cheddar.ciphertext) -> !cheddar.ciphertext {
  // CHECK-NOT: cheddar.hrot %
  // CHECK-NOT: cheddar.add
  // CHECK: cheddar.hrot_add
  // CHECK-SAME: distance = 3
  %rotated = cheddar.hrot %ctx, %ct0 {static_distance = 3 : i64} : (!cheddar.context, !cheddar.ciphertext) -> !cheddar.ciphertext
  %sum = cheddar.add %ctx, %rotated, %ct1 : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
  return %sum : !cheddar.ciphertext
}

// hrot (constant dynamic distance, the rotate-and-sum form) + add -> hrot_add
// CHECK: @fuse_hrot_add_dynamic_const
func.func @fuse_hrot_add_dynamic_const(
    %ctx: !cheddar.context, %ct0: !cheddar.ciphertext,
    %ct1: !cheddar.ciphertext) -> !cheddar.ciphertext {
  // CHECK-NOT: cheddar.hrot %
  // CHECK-NOT: cheddar.add
  // CHECK: cheddar.hrot_add
  // CHECK-SAME: distance = 5
  %d = arith.constant 5 : index
  %rotated = cheddar.hrot %ctx, %ct0, %d : (!cheddar.context, !cheddar.ciphertext, index) -> !cheddar.ciphertext
  %sum = cheddar.add %ctx, %rotated, %ct1 : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
  return %sum : !cheddar.ciphertext
}

// hconj + add -> hconj_add
// CHECK: @fuse_hconj_add
func.func @fuse_hconj_add(
    %ctx: !cheddar.context, %ct0: !cheddar.ciphertext,
    %ct1: !cheddar.ciphertext) -> !cheddar.ciphertext {
  // CHECK-NOT: cheddar.hconj %
  // CHECK-NOT: cheddar.add
  // CHECK: cheddar.hconj_add
  %conj = cheddar.hconj %ctx, %ct0 : (!cheddar.context, !cheddar.ciphertext) -> !cheddar.ciphertext
  %sum = cheddar.add %ctx, %conj, %ct1 : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
  return %sum : !cheddar.ciphertext
}

// A non-constant dynamic distance cannot be folded into hrot_add's static
// attribute, so the hrot + add is left unfused.
// CHECK: @no_fuse_hrot_add_nonconst
func.func @no_fuse_hrot_add_nonconst(
    %ctx: !cheddar.context, %ct0: !cheddar.ciphertext,
    %ct1: !cheddar.ciphertext, %d: index) -> !cheddar.ciphertext {
  // CHECK: cheddar.hrot
  // CHECK: cheddar.add
  // CHECK-NOT: cheddar.hrot_add
  %rotated = cheddar.hrot %ctx, %ct0, %d : (!cheddar.context, !cheddar.ciphertext, index) -> !cheddar.ciphertext
  %sum = cheddar.add %ctx, %rotated, %ct1 : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
  return %sum : !cheddar.ciphertext
}

// A sub_plain scheduled between mult and relinearize (a degree-3 ciphertext in
// the sub) is hoisted: relin moves before sub_plain, which re-exposes the
// mult+relin -> hmult fusion. Result: hmult then sub_plain, no mult/relin left.
// CHECK: @hoist_relin_before_sub_plain
func.func @hoist_relin_before_sub_plain(
    %ctx: !cheddar.context, %ct0: !cheddar.ciphertext,
    %ct1: !cheddar.ciphertext, %pt: !cheddar.plaintext,
    %key: !cheddar.eval_key) -> !cheddar.ciphertext {
  // CHECK-NOT: cheddar.mult
  // CHECK-NOT: cheddar.relinearize
  // CHECK: cheddar.hmult
  // CHECK: cheddar.sub_plain
  %mult = cheddar.mult %ctx, %ct0, %ct1 : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
  %sub = cheddar.sub_plain %ctx, %mult, %pt : (!cheddar.context, !cheddar.ciphertext, !cheddar.plaintext) -> !cheddar.ciphertext
  %relin = cheddar.relinearize %ctx, %sub, %key : (!cheddar.context, !cheddar.ciphertext, !cheddar.eval_key) -> !cheddar.ciphertext
  return %relin : !cheddar.ciphertext
}
