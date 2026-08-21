// RUN: heir-opt --cheddar-bufferize --fold-memref-alias-ops %s | FileCheck %s

// The generic Cheddar DPS model must turn tensor payloads into buffers, erase
// the tied SSA result, and replace uses with the destination buffer. Cover both
// the usual trailing destination and the ops whose in-place destination is in a
// nonstandard operand position.

// CHECK: func.func @add
// CHECK: cheddar.add {{.*}} : (!context, memref<!ciphertext>, memref<!ciphertext>, memref<!ciphertext>) -> ()
// CHECK: return
func.func @add(%ctx: !cheddar.context, %lhs: tensor<!cheddar.ciphertext>, %rhs: tensor<!cheddar.ciphertext>, %out: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
  %result = cheddar.add %ctx, %lhs, %rhs, %out : (!cheddar.context, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  return %result : tensor<!cheddar.ciphertext>
}

// CHECK: func.func @mad_unsafe
// CHECK: cheddar.mad_unsafe {{.*}} : (!context, memref<!ciphertext>, memref<!ciphertext>, memref<!constant>) -> ()
// CHECK: return
func.func @mad_unsafe(%ctx: !cheddar.context, %acc: tensor<!cheddar.ciphertext>, %input: tensor<!cheddar.ciphertext>, %constant: tensor<!cheddar.constant>) -> tensor<!cheddar.ciphertext> {
  %result = cheddar.mad_unsafe %ctx, %acc, %input, %constant : (!cheddar.context, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>, tensor<!cheddar.constant>) -> tensor<!cheddar.ciphertext>
  return %result : tensor<!cheddar.ciphertext>
}

// CHECK: func.func @prepare_rot_key
// CHECK: cheddar.prepare_rot_key %{{.*}} {distance = 7 : i64, maxLevel = 13 : i64} : (memref<!user_interface>) -> ()
// CHECK: return
func.func @prepare_rot_key(%ui: tensor<!cheddar.user_interface>) -> tensor<!cheddar.user_interface> {
  %result = cheddar.prepare_rot_key %ui {distance = 7 : i64, maxLevel = 13 : i64} : (tensor<!cheddar.user_interface>) -> tensor<!cheddar.user_interface>
  return %result : tensor<!cheddar.user_interface>
}

// Both objects mutated by bootstrap setup are DPS destinations/results.
// CHECK: func.func @prepare_bootstrap
// CHECK: cheddar.prepare_bootstrap %{{.*}}, %{{.*}} {numSlots = 8 : i64} : (memref<!boot_context>, memref<!user_interface>) -> ()
// CHECK: return
func.func @prepare_bootstrap(%ctx: tensor<!cheddar.boot_context>, %ui: tensor<!cheddar.user_interface>) -> (tensor<!cheddar.boot_context>, tensor<!cheddar.user_interface>) {
  %new_ctx, %new_ui = cheddar.prepare_bootstrap %ctx, %ui {numSlots = 8 : i64} : (tensor<!cheddar.boot_context>, tensor<!cheddar.user_interface>) -> (tensor<!cheddar.boot_context>, tensor<!cheddar.user_interface>)
  return %new_ctx, %new_ui : tensor<!cheddar.boot_context>, tensor<!cheddar.user_interface>
}

// CHECK: func.func @decode
// CHECK-SAME: %[[DECODED:[a-zA-Z0-9_]+]]: memref<4xf64>
// CHECK: cheddar.decode %{{.*}}, %{{.*}}, %[[DECODED]] : (!encoder, memref<!plaintext>, memref<4xf64>) -> ()
// CHECK: return{{$}}
func.func @decode(%encoder: !cheddar.encoder, %plaintext: tensor<!cheddar.plaintext>, %value: tensor<4xf64>) -> tensor<4xf64> {
  %decoded = cheddar.decode %encoder, %plaintext, %value : (!cheddar.encoder, tensor<!cheddar.plaintext>, tensor<4xf64>) -> tensor<4xf64>
  return %decoded : tensor<4xf64>
}

// A chain of fully-overwriting arithmetic operations can keep using one
// destination when the previous value dies. This is the normal DPS/One-Shot
// path; no Cheddar-specific post-bufferization rewrite is involved.
// CHECK: func.func @reuse_sequence
// CHECK-SAME: %[[STORAGE:[a-zA-Z0-9_]+]]: memref<!ciphertext> {bufferize.result}
// CHECK-NOT: memref.alloc
// CHECK: cheddar.add %{{.*}}, %{{.*}}, %{{.*}}, %[[STORAGE]]
// CHECK: cheddar.neg %{{.*}}, %[[STORAGE]], %[[STORAGE]]
// CHECK-NOT: memref.copy
func.func @reuse_sequence(%ctx: !cheddar.context, %lhs: tensor<!cheddar.ciphertext>, %rhs: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
  %empty = tensor.empty() : tensor<!cheddar.ciphertext>
  %sum = cheddar.add %ctx, %lhs, %rhs, %empty : (!cheddar.context, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  %negated = cheddar.neg %ctx, %sum, %sum : (!cheddar.context, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  return %negated : tensor<!cheddar.ciphertext>
}

// A shape-only tensor.empty still requests a distinct destination. Alias
// support does not force reuse unless the lowering explicitly chooses it.
// CHECK: func.func @rescale_fresh
// CHECK-SAME: %[[OUTPUT:[a-zA-Z0-9_]+]]: memref<!ciphertext> {bufferize.result}
// CHECK: %[[INPUT:[a-zA-Z0-9_]+]] = memref.alloc(){{.*}} : memref<!ciphertext>
// CHECK: cheddar.add %{{.*}}, %{{.*}}, %{{.*}}, %[[INPUT]]
// CHECK-NOT: memref.alloc
// CHECK: cheddar.rescale %{{.*}}, %[[INPUT]], %[[OUTPUT]]
func.func @rescale_fresh(%ctx: !cheddar.context, %lhs: tensor<!cheddar.ciphertext>, %rhs: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
  %inputEmpty = tensor.empty() : tensor<!cheddar.ciphertext>
  %input = cheddar.add %ctx, %lhs, %rhs, %inputEmpty : (!cheddar.context, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  %outputEmpty = tensor.empty() : tensor<!cheddar.ciphertext>
  %output = cheddar.rescale %ctx, %input, %outputEmpty : (!cheddar.context, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  return %output : tensor<!cheddar.ciphertext>
}

// scale-snu implements an aliasing Rescale by staging through an internal
// temporary and moving it back, so an explicitly reused destination remains
// the same buffer from One-Shot's perspective.
// CHECK: func.func @rescale_alias_hint
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: memref<!ciphertext> {bufferize.result}
// CHECK-NOT: memref.alloc
// CHECK: cheddar.add %{{.*}}, %{{.*}}, %{{.*}}, %[[INPUT]]
// CHECK: cheddar.rescale %{{.*}}, %[[INPUT]], %[[INPUT]]
// CHECK-NOT: memref.copy
func.func @rescale_alias_hint(%ctx: !cheddar.context, %lhs: tensor<!cheddar.ciphertext>, %rhs: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
  %empty = tensor.empty() : tensor<!cheddar.ciphertext>
  %input = cheddar.add %ctx, %lhs, %rhs, %empty : (!cheddar.context, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  %output = cheddar.rescale %ctx, %input, %input : (!cheddar.context, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  return %output : tensor<!cheddar.ciphertext>
}

// scale-snu explicitly supports the fused rotation-add with all three
// ciphertext references aliasing, so One-Shot can keep the same destination.
// CHECK: func.func @hrot_add_in_place
// CHECK-SAME: %[[STORAGE:[a-zA-Z0-9_]+]]: memref<!ciphertext> {bufferize.result}
// CHECK-NOT: memref.alloc
// CHECK: cheddar.add %{{.*}}, %{{.*}}, %{{.*}}, %[[STORAGE]]
// CHECK: cheddar.hrot_add %{{.*}}, %{{.*}}, %[[STORAGE]], %[[STORAGE]], %[[STORAGE]]
// CHECK-NOT: memref.copy
func.func @hrot_add_in_place(%ctx: !cheddar.context, %ui: !cheddar.user_interface, %lhs: tensor<!cheddar.ciphertext>, %rhs: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
  %empty = tensor.empty() : tensor<!cheddar.ciphertext>
  %input = cheddar.add %ctx, %lhs, %rhs, %empty : (!cheddar.context, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  %output = cheddar.hrot_add %ctx, %ui, %input, %input, %input {distance = 2 : i64} : (!cheddar.context, !cheddar.user_interface, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  return %output : tensor<!cheddar.ciphertext>
}

// Region-carried values stay equivalent to their iter_args. This exercises
// the stock One-Shot loop analysis; Cheddar never rewrites the destination
// after analysis.
// CHECK: func.func @loop_in_place
// The loop starts from a caller-owned input, not a shape-only tensor.empty, so
// preserving functional input semantics requires one real boundary copy.
// CHECK: memref.copy %{{.*}}, %{{.*}} : memref<!ciphertext> to memref<!ciphertext>
// CHECK: scf.for {{.*}} iter_args(%[[ITER:[a-zA-Z0-9_]+]] = %{{.*}}) -> (memref<!ciphertext>) {
// CHECK: cheddar.neg %{{.*}}, %[[ITER]], %[[ITER]]
// CHECK: scf.yield %[[ITER]] : memref<!ciphertext>
func.func @loop_in_place(%ctx: !cheddar.context, %init: tensor<!cheddar.ciphertext>, %upper: index) -> tensor<!cheddar.ciphertext> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %result = scf.for %i = %c0 to %upper step %c1 iter_args(%iter = %init) -> tensor<!cheddar.ciphertext> {
    %next = cheddar.neg %ctx, %iter, %iter : (!cheddar.context, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
    scf.yield %next : tensor<!cheddar.ciphertext>
  }
  return %result : tensor<!cheddar.ciphertext>
}
