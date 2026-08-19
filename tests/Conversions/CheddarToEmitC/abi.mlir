// RUN: heir-opt --convert-to-emitc=filter-dialects=cheddar --cheddar-emitc-boundary --reconcile-unrealized-casts %s | FileCheck %s

// The ABI substrate is independently useful without lowering a CHEDDAR
// runtime operation: move-only payload buffers become C++ references, and
// cross-function calls remain structured.

// CHECK: func.func @abi_inner(
// CHECK-SAME: !emitc.opaque<"const std::array<Ciphertext<word>, 1>&">
// CHECK-SAME: !emitc.opaque<"std::array<Ciphertext<word>, 1>&">
func.func @abi_inner(
    %input: memref<1x!cheddar.ciphertext>,
    %output: memref<1x!cheddar.ciphertext> {bufferize.result}) {
  return
}

// CHECK: func.func @abi_outer(
// CHECK-SAME: !emitc.opaque<"const std::array<Ciphertext<word>, 1>&">
// CHECK-SAME: !emitc.opaque<"std::array<Ciphertext<word>, 1>&">
// CHECK: emitc.call_opaque "abi_inner"
func.func @abi_outer(
    %input: memref<1x!cheddar.ciphertext>,
    %output: memref<1x!cheddar.ciphertext>) {
  func.call @abi_inner(%input, %output)
      : (memref<1x!cheddar.ciphertext>, memref<1x!cheddar.ciphertext>) -> ()
  return
}

// Mutability propagates to a fixed point through more than one call edge.
// CHECK: func.func @abi_outermost(
// CHECK-SAME: !emitc.opaque<"const std::array<Ciphertext<word>, 1>&">
// CHECK-SAME: !emitc.opaque<"std::array<Ciphertext<word>, 1>&">
// CHECK: emitc.call_opaque "abi_outer"
func.func @abi_outermost(
    %input: memref<1x!cheddar.ciphertext>,
    %output: memref<1x!cheddar.ciphertext>) {
  func.call @abi_outer(%input, %output)
      : (memref<1x!cheddar.ciphertext>, memref<1x!cheddar.ciphertext>) -> ()
  return
}

// Rewriting a call to a refified function preserves ordinary scalar results.
// CHECK: func.func @abi_result_inner
func.func @abi_result_inner(
    %value: i32,
    %output: memref<1x!cheddar.ciphertext> {bufferize.result}) -> i32 {
  return %value : i32
}

// CHECK: func.func @abi_result_outer
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = emitc.call_opaque "abi_result_inner"
// CHECK: return %[[RESULT]] : i32
func.func @abi_result_outer(
    %value: i32,
    %output: memref<1x!cheddar.ciphertext>) -> i32 {
  %result = func.call @abi_result_inner(%value, %output)
      : (i32, memref<1x!cheddar.ciphertext>) -> i32
  return %result : i32
}
