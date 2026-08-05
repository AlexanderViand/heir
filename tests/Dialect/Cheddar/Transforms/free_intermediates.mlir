// RUN: heir-opt --cheddar-free-intermediates %s | FileCheck %s

!ciphertext = !cheddar.ciphertext
!context = !cheddar.context
!eval_key = !cheddar.eval_key
!plaintext = !cheddar.plaintext

// The local payload intermediate (%alloc, the mult result) is freed right after
// its last use (the relinearize), so its GPU buffer is released at last use
// rather than at function-scope cleanup. The returned out-param (%arg2) is a
// function argument, not a memref.alloc, so it is never freed.

// CHECK: @free_intermediate
// CHECK: %[[A:.*]] = memref.alloc
// CHECK: cheddar.mult %{{.*}}, %[[A]]
// CHECK: cheddar.relinearize %{{.*}}, %[[A]]
// CHECK-NEXT: memref.dealloc %[[A]]
// CHECK-NOT: memref.dealloc %arg2
func.func @free_intermediate(%ctx: !context, %arg0: memref<!ciphertext>,
    %arg1: memref<!ciphertext>, %evk: !eval_key,
    %arg2: memref<!ciphertext> {bufferize.result}) {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<!ciphertext>
  cheddar.mult %ctx, %arg0, %arg1, %alloc : (!context, memref<!ciphertext>, memref<!ciphertext>, memref<!ciphertext>) -> ()
  cheddar.relinearize %ctx, %alloc, %evk, %arg2 : (!context, memref<!ciphertext>, !eval_key, memref<!ciphertext>) -> ()
  return
}

// A plain float buffer is a scope-bound stack array with nothing to free: the
// pass leaves it alone (only CHEDDAR payload buffers are freed).
// CHECK: @no_free_for_float
// CHECK-NOT: memref.dealloc
func.func @no_free_for_float(%arg0: memref<8xf32>) {
  %alloc = memref.alloc() : memref<8xf32>
  %c0 = arith.constant 0 : index
  %v = memref.load %alloc[%c0] : memref<8xf32>
  memref.store %v, %arg0[%c0] : memref<8xf32>
  return
}

// A payload staged out of a read-only buffer (the caller-owned
// split-preprocessing storage argument) is forwarded: the reader uses the
// storage slot directly and the staging alloc disappears. Without this, the
// staging store lowers to `local = std::move(storage[i])` -- consuming the
// caller's plaintext, so a second call over the same storage reads moved-from
// payloads. Nothing is freed: the slot belongs to the caller.
// CHECK: @forward_arg_staging
// CHECK-NOT: memref.alloc
// CHECK: %[[SV:.*]] = memref.subview %{{.*}}[%{{.*}}] [1] [1]
// CHECK: cheddar.mult_plain %{{.*}}, %{{.*}}, %[[SV]], %{{.*}} :
// CHECK-NOT: memref.dealloc
func.func @forward_arg_staging(%ctx: !context, %storage: memref<8x!plaintext>,
    %ct: memref<!ciphertext>, %i: index,
    %out: memref<!ciphertext> {bufferize.result}) {
  %v = memref.load %storage[%i] : memref<8x!plaintext>
  %staged = memref.alloc() : memref<!plaintext>
  memref.store %v, %staged[] : memref<!plaintext>
  cheddar.mult_plain %ctx, %ct, %staged, %out : (!context, memref<!ciphertext>, memref<!plaintext>, memref<!ciphertext>) -> ()
  return
}

// The common fully-unrolled shape: a constant staging index. Forwarded the
// same way (the later canonicalize may fold the offset static, in which case
// the identity-layout cast is skipped -- the readers accept any layout).
// CHECK: @forward_arg_staging_const_index
// CHECK-NOT: memref.alloc
// CHECK: %[[SV:.*]] = memref.subview %{{.*}}[%{{.*}}] [1] [1]
// CHECK: cheddar.mult_plain
// CHECK-NOT: memref.dealloc
func.func @forward_arg_staging_const_index(%ctx: !context,
    %storage: memref<8x!plaintext>, %ct: memref<!ciphertext>,
    %out: memref<!ciphertext> {bufferize.result}) {
  %c1 = arith.constant 1 : index
  %v = memref.load %storage[%c1] : memref<8x!plaintext>
  %staged = memref.alloc() : memref<!plaintext>
  memref.store %v, %staged[] : memref<!plaintext>
  cheddar.mult_plain %ctx, %ct, %staged, %out : (!context, memref<!ciphertext>, memref<!plaintext>, memref<!ciphertext>) -> ()
  return
}

// Staging out of a buffer this function writes (a locally filled storage) is
// NOT forwarded -- the source element may be overwritten between the staging
// copy and its use -- so the staging alloc stays and is freed at last use.
// CHECK: @keep_staging_of_written_buffer
// CHECK: %[[ST:.*]] = memref.alloc() : memref<!plaintext>
// CHECK: cheddar.mult_plain %{{.*}}, %{{.*}}, %[[ST]]
// CHECK-NEXT: memref.dealloc %[[ST]]
func.func @keep_staging_of_written_buffer(%ctx: !context,
    %storage: memref<8x!plaintext>, %pt: memref<8x!plaintext>,
    %ct: memref<!ciphertext>, %i: index,
    %out: memref<!ciphertext> {bufferize.result}) {
  %w = memref.load %pt[%i] : memref<8x!plaintext>
  memref.store %w, %storage[%i] : memref<8x!plaintext>
  %v = memref.load %storage[%i] : memref<8x!plaintext>
  %staged = memref.alloc() : memref<!plaintext>
  memref.store %v, %staged[] : memref<!plaintext>
  cheddar.mult_plain %ctx, %ct, %staged, %out : (!context, memref<!ciphertext>, memref<!plaintext>, memref<!ciphertext>) -> ()
  return
}
