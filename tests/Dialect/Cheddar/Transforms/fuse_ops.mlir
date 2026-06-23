// RUN: heir-opt --cheddar-fuse-ops %s | FileCheck %s

!ct = !cheddar.ciphertext

// mult + relinearize + rescale -> hmult (rescale=true, elided as the default)
// CHECK: @fuse_hmult_rescale
func.func @fuse_hmult_rescale(
    %ctx: !cheddar.context, %ct0: tensor<!ct>,
    %ct1: tensor<!ct>, %key: !cheddar.eval_key) -> tensor<!ct> {
  // CHECK-NOT: cheddar.mult
  // CHECK-NOT: cheddar.relinearize
  // CHECK-NOT: cheddar.rescale
  // CHECK: cheddar.hmult
  // CHECK-NOT: rescale = false
  %d0 = bufferization.alloc_tensor() : tensor<!ct>
  %mult = cheddar.mult %ctx, %ct0, %ct1, %d0 : (!cheddar.context, tensor<!ct>, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  %d1 = bufferization.alloc_tensor() : tensor<!ct>
  %relin = cheddar.relinearize %ctx, %mult, %key, %d1 : (!cheddar.context, tensor<!ct>, !cheddar.eval_key, tensor<!ct>) -> tensor<!ct>
  %d2 = bufferization.alloc_tensor() : tensor<!ct>
  %rescaled = cheddar.rescale %ctx, %relin, %d2 : (!cheddar.context, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  return %rescaled : tensor<!ct>
}

// mult + relinearize -> hmult (rescale=false)
// CHECK: @fuse_hmult_no_rescale
func.func @fuse_hmult_no_rescale(
    %ctx: !cheddar.context, %ct0: tensor<!ct>,
    %ct1: tensor<!ct>, %key: !cheddar.eval_key) -> tensor<!ct> {
  // CHECK-NOT: cheddar.mult
  // CHECK-NOT: cheddar.relinearize
  // CHECK: cheddar.hmult
  // CHECK-SAME: rescale = false
  %d0 = bufferization.alloc_tensor() : tensor<!ct>
  %mult = cheddar.mult %ctx, %ct0, %ct1, %d0 : (!cheddar.context, tensor<!ct>, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  %d1 = bufferization.alloc_tensor() : tensor<!ct>
  %relin = cheddar.relinearize %ctx, %mult, %key, %d1 : (!cheddar.context, tensor<!ct>, !cheddar.eval_key, tensor<!ct>) -> tensor<!ct>
  return %relin : tensor<!ct>
}

// mult + relinearize_rescale -> hmult (rescale=true)
// CHECK: @fuse_hmult_relin_rescale
func.func @fuse_hmult_relin_rescale(
    %ctx: !cheddar.context, %ct0: tensor<!ct>,
    %ct1: tensor<!ct>, %key: !cheddar.eval_key) -> tensor<!ct> {
  // CHECK-NOT: cheddar.mult
  // CHECK-NOT: cheddar.relinearize_rescale
  // CHECK: cheddar.hmult
  %d0 = bufferization.alloc_tensor() : tensor<!ct>
  %mult = cheddar.mult %ctx, %ct0, %ct1, %d0 : (!cheddar.context, tensor<!ct>, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  %d1 = bufferization.alloc_tensor() : tensor<!ct>
  %rr = cheddar.relinearize_rescale %ctx, %mult, %key, %d1 : (!cheddar.context, tensor<!ct>, !cheddar.eval_key, tensor<!ct>) -> tensor<!ct>
  return %rr : tensor<!ct>
}

// hrot (static distance) + add -> hrot_add
// CHECK: @fuse_hrot_add_static
func.func @fuse_hrot_add_static(
    %ctx: !cheddar.context, %ct0: tensor<!ct>,
    %ct1: tensor<!ct>) -> tensor<!ct> {
  // CHECK-NOT: cheddar.hrot %
  // CHECK-NOT: cheddar.add
  // CHECK: cheddar.hrot_add
  // CHECK-SAME: distance = 3
  %d0 = bufferization.alloc_tensor() : tensor<!ct>
  %rotated = cheddar.hrot %ctx, %ct0, %d0 {static_distance = 3 : i64} : (!cheddar.context, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  %d1 = bufferization.alloc_tensor() : tensor<!ct>
  %sum = cheddar.add %ctx, %rotated, %ct1, %d1 : (!cheddar.context, tensor<!ct>, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  return %sum : tensor<!ct>
}

// hrot (constant dynamic distance, the rotate-and-sum form) + add -> hrot_add
// CHECK: @fuse_hrot_add_dynamic_const
func.func @fuse_hrot_add_dynamic_const(
    %ctx: !cheddar.context, %ct0: tensor<!ct>,
    %ct1: tensor<!ct>) -> tensor<!ct> {
  // CHECK-NOT: cheddar.hrot %
  // CHECK-NOT: cheddar.add
  // CHECK: cheddar.hrot_add
  // CHECK-SAME: distance = 5
  %d = arith.constant 5 : index
  %d0 = bufferization.alloc_tensor() : tensor<!ct>
  %rotated = cheddar.hrot %ctx, %ct0, %d0, %d : (!cheddar.context, tensor<!ct>, tensor<!ct>, index) -> tensor<!ct>
  %d1 = bufferization.alloc_tensor() : tensor<!ct>
  %sum = cheddar.add %ctx, %rotated, %ct1, %d1 : (!cheddar.context, tensor<!ct>, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  return %sum : tensor<!ct>
}

// hconj + add -> hconj_add
// CHECK: @fuse_hconj_add
func.func @fuse_hconj_add(
    %ctx: !cheddar.context, %ct0: tensor<!ct>,
    %ct1: tensor<!ct>) -> tensor<!ct> {
  // CHECK-NOT: cheddar.hconj %
  // CHECK-NOT: cheddar.add
  // CHECK: cheddar.hconj_add
  %d0 = bufferization.alloc_tensor() : tensor<!ct>
  %conj = cheddar.hconj %ctx, %ct0, %d0 : (!cheddar.context, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  %d1 = bufferization.alloc_tensor() : tensor<!ct>
  %sum = cheddar.add %ctx, %conj, %ct1, %d1 : (!cheddar.context, tensor<!ct>, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  return %sum : tensor<!ct>
}

// A non-constant dynamic distance cannot be folded into hrot_add's static
// attribute, so the hrot + add is left unfused.
// CHECK: @no_fuse_hrot_add_nonconst
func.func @no_fuse_hrot_add_nonconst(
    %ctx: !cheddar.context, %ct0: tensor<!ct>,
    %ct1: tensor<!ct>, %d: index) -> tensor<!ct> {
  // CHECK: cheddar.hrot
  // CHECK: cheddar.add
  // CHECK-NOT: cheddar.hrot_add
  %d0 = bufferization.alloc_tensor() : tensor<!ct>
  %rotated = cheddar.hrot %ctx, %ct0, %d0, %d : (!cheddar.context, tensor<!ct>, tensor<!ct>, index) -> tensor<!ct>
  %d1 = bufferization.alloc_tensor() : tensor<!ct>
  %sum = cheddar.add %ctx, %rotated, %ct1, %d1 : (!cheddar.context, tensor<!ct>, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  return %sum : tensor<!ct>
}

// A sub_plain scheduled between mult and relinearize (a degree-3 ciphertext in
// the sub) is hoisted: relin moves before sub_plain, which re-exposes the
// mult+relin -> hmult fusion. Result: hmult then sub_plain, no mult/relin left.
// CHECK: @hoist_relin_before_sub_plain
func.func @hoist_relin_before_sub_plain(
    %ctx: !cheddar.context, %ct0: tensor<!ct>,
    %ct1: tensor<!ct>, %pt: tensor<!cheddar.plaintext>,
    %key: !cheddar.eval_key) -> tensor<!ct> {
  // CHECK-NOT: cheddar.mult
  // CHECK-NOT: cheddar.relinearize
  // CHECK: cheddar.hmult
  // CHECK: cheddar.sub_plain
  %d0 = bufferization.alloc_tensor() : tensor<!ct>
  %mult = cheddar.mult %ctx, %ct0, %ct1, %d0 : (!cheddar.context, tensor<!ct>, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  %d1 = bufferization.alloc_tensor() : tensor<!ct>
  %sub = cheddar.sub_plain %ctx, %mult, %pt, %d1 : (!cheddar.context, tensor<!ct>, tensor<!cheddar.plaintext>, tensor<!ct>) -> tensor<!ct>
  %d2 = bufferization.alloc_tensor() : tensor<!ct>
  %relin = cheddar.relinearize %ctx, %sub, %key, %d2 : (!cheddar.context, tensor<!ct>, !cheddar.eval_key, tensor<!ct>) -> tensor<!ct>
  return %relin : tensor<!ct>
}
