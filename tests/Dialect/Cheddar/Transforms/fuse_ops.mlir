// RUN: heir-opt --cheddar-fuse-ops %s | FileCheck %s

!ct = !cheddar.ciphertext

// CHECK: @fuse_hmult_rescale
func.func @fuse_hmult_rescale(
    %ctx: !cheddar.context, %lhs: tensor<!ct>, %rhs: tensor<!ct>,
    %key: !cheddar.eval_key) -> tensor<!ct> {
  // CHECK-NOT: cheddar.mult
  // CHECK-NOT: cheddar.relinearize
  // CHECK-NOT: cheddar.rescale
  // CHECK: cheddar.hmult
  // CHECK-NOT: rescale = false
  %d0 = bufferization.alloc_tensor() : tensor<!ct>
  %mult = cheddar.mult %ctx, %lhs, %rhs, %d0 : (!cheddar.context, tensor<!ct>, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  %d1 = bufferization.alloc_tensor() : tensor<!ct>
  %relin = cheddar.relinearize %ctx, %mult, %key, %d1 : (!cheddar.context, tensor<!ct>, !cheddar.eval_key, tensor<!ct>) -> tensor<!ct>
  %d2 = bufferization.alloc_tensor() : tensor<!ct>
  %result = cheddar.rescale %ctx, %relin, %d2 : (!cheddar.context, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  return %result : tensor<!ct>
}

// CHECK: @fuse_hmult_no_rescale
func.func @fuse_hmult_no_rescale(
    %ctx: !cheddar.context, %lhs: tensor<!ct>, %rhs: tensor<!ct>,
    %key: !cheddar.eval_key) -> tensor<!ct> {
  // CHECK-NOT: cheddar.mult
  // CHECK-NOT: cheddar.relinearize
  // CHECK: cheddar.hmult
  // CHECK-SAME: rescale = false
  %d0 = bufferization.alloc_tensor() : tensor<!ct>
  %mult = cheddar.mult %ctx, %lhs, %rhs, %d0 : (!cheddar.context, tensor<!ct>, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  %d1 = bufferization.alloc_tensor() : tensor<!ct>
  %result = cheddar.relinearize %ctx, %mult, %key, %d1 : (!cheddar.context, tensor<!ct>, !cheddar.eval_key, tensor<!ct>) -> tensor<!ct>
  return %result : tensor<!ct>
}

// CHECK: @fuse_hmult_relinearize_rescale
func.func @fuse_hmult_relinearize_rescale(
    %ctx: !cheddar.context, %lhs: tensor<!ct>, %rhs: tensor<!ct>,
    %key: !cheddar.eval_key) -> tensor<!ct> {
  // CHECK-NOT: cheddar.mult
  // CHECK-NOT: cheddar.relinearize_rescale
  // CHECK: cheddar.hmult
  %d0 = bufferization.alloc_tensor() : tensor<!ct>
  %mult = cheddar.mult %ctx, %lhs, %rhs, %d0 : (!cheddar.context, tensor<!ct>, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  %d1 = bufferization.alloc_tensor() : tensor<!ct>
  %result = cheddar.relinearize_rescale %ctx, %mult, %key, %d1 : (!cheddar.context, tensor<!ct>, !cheddar.eval_key, tensor<!ct>) -> tensor<!ct>
  return %result : tensor<!ct>
}

// CHECK: @fuse_rotation_and_conjugation
func.func @fuse_rotation_and_conjugation(
    %ctx: !cheddar.context, %evk: !cheddar.evk_map,
    %input: tensor<!ct>, %other: tensor<!ct>) -> (tensor<!ct>, tensor<!ct>) {
  // CHECK: cheddar.hrot_add
  // CHECK-SAME: distance = 3
  // CHECK-SAME: level = 3
  // CHECK: cheddar.hconj_add
  %r0 = bufferization.alloc_tensor() : tensor<!ct>
  %rotated = cheddar.hrot %ctx, %evk, %input, %r0 {level = 3 : i64, static_distance = 3 : i64} : (!cheddar.context, !cheddar.evk_map, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  %r1 = bufferization.alloc_tensor() : tensor<!ct>
  %rotated_sum = cheddar.add %ctx, %rotated, %other, %r1 : (!cheddar.context, tensor<!ct>, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  %c0 = bufferization.alloc_tensor() : tensor<!ct>
  %conjugated = cheddar.hconj %ctx, %evk, %input, %c0 : (!cheddar.context, !cheddar.evk_map, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  %c1 = bufferization.alloc_tensor() : tensor<!ct>
  %conjugated_sum = cheddar.add %ctx, %conjugated, %other, %c1 : (!cheddar.context, tensor<!ct>, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  return %rotated_sum, %conjugated_sum : tensor<!ct>, tensor<!ct>
}

// Relinearization commutes with add_plain when all operations share a context.
// CHECK: @hoist_relinearize
func.func @hoist_relinearize(
    %ctx: !cheddar.context, %lhs: tensor<!ct>, %rhs: tensor<!ct>,
    %plain: tensor<!cheddar.plaintext>, %key: !cheddar.eval_key) -> tensor<!ct> {
  // CHECK-NOT: cheddar.mult
  // CHECK-NOT: cheddar.relinearize
  // CHECK: cheddar.hmult
  // CHECK: cheddar.add_plain
  %d0 = bufferization.alloc_tensor() : tensor<!ct>
  %mult = cheddar.mult %ctx, %lhs, %rhs, %d0 : (!cheddar.context, tensor<!ct>, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  %d1 = bufferization.alloc_tensor() : tensor<!ct>
  %added = cheddar.add_plain %ctx, %mult, %plain, %d1 : (!cheddar.context, tensor<!ct>, tensor<!cheddar.plaintext>, tensor<!ct>) -> tensor<!ct>
  %d2 = bufferization.alloc_tensor() : tensor<!ct>
  %result = cheddar.relinearize %ctx, %added, %key, %d2 : (!cheddar.context, tensor<!ct>, !cheddar.eval_key, tensor<!ct>) -> tensor<!ct>
  return %result : tensor<!ct>
}

// Fusing across different contexts would silently change semantics.
// CHECK: @do_not_fuse_different_contexts
func.func @do_not_fuse_different_contexts(
    %ctx0: !cheddar.context, %ctx1: !cheddar.context,
    %evk: !cheddar.evk_map, %input: tensor<!ct>,
    %other: tensor<!ct>) -> tensor<!ct> {
  // CHECK: cheddar.hrot
  // CHECK: cheddar.add
  // CHECK-NOT: cheddar.hrot_add
  %d0 = bufferization.alloc_tensor() : tensor<!ct>
  %rotated = cheddar.hrot %ctx0, %evk, %input, %d0 {level = 2 : i64, static_distance = 2 : i64} : (!cheddar.context, !cheddar.evk_map, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  %d1 = bufferization.alloc_tensor() : tensor<!ct>
  %result = cheddar.add %ctx1, %rotated, %other, %d1 : (!cheddar.context, tensor<!ct>, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  return %result : tensor<!ct>
}
