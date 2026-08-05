// RUN: heir-opt "--one-shot-bufferize=bufferize-function-boundaries=true function-boundary-type-conversion=identity-layout-map" "--buffer-results-to-out-params=hoist-static-allocs=true modify-public-functions=true add-result-attr=true" --fold-memref-alias-ops --canonicalize --convert-to-emitc --cheddar-emitc-boundary --reconcile-unrealized-casts %s | FileCheck %s

// A `__heir_debug_*` call (the form LWEToCheddar produces for cheddar --debug)
// lowers to a free C++ `__heir_debug(encoder, ui, ct, "name", "metadata")`
// call. The external `func.func` declaration is erased (the upstream Cpp
// emitter cannot print an external func.func), so medusa's C++ prelude defines
// `__heir_debug`.

!ciphertext = !cheddar.ciphertext
!context = !cheddar.context
!encoder = !cheddar.encoder
!user_interface = !cheddar.user_interface

// The external declaration must NOT survive into the emitted module.
// CHECK-NOT: @__heir_debug_0
func.func private @__heir_debug_0(!encoder, !user_interface, tensor<!ciphertext>)

// CHECK: func.func @debug_chain
// Encoder + scalar ciphertext inputs are tightened to const C++ references; the
// UserInterface stays a pointer.
// CHECK-SAME: !emitc.opaque<"const Encoder<word>&">
// CHECK-SAME: !emitc.ptr<!emitc.opaque<"UserInterface<word>">>
// CHECK-SAME: !emitc.opaque<"const Ciphertext<word>&">
// The debug.validate -> __heir_debug call: encoder, ui, ct operands plus the
// name + metadata baked as trailing string-literal opaque args.
// CHECK: emitc.call_opaque "__heir_debug"
// CHECK-SAME: %arg0, %arg1, %arg2
// CHECK-SAME: heir_val0
// CHECK-SAME: heir_meta0
func.func @debug_chain(%enc: !encoder, %ui: !user_interface, %ct: tensor<!ciphertext>) {
  func.call @__heir_debug_0(%enc, %ui, %ct) {debug.name = "heir_val0", debug.metadata = "heir_meta0"} : (!encoder, !user_interface, tensor<!ciphertext>) -> ()
  return
}

// A rank-1 (1-element array) ciphertext value -- the usual cheddar value rep --
// lowers the ct operand to a const std::array<Ciphertext<word>, N>& (the medusa
// C++ hook should accept both this and the scalar form, e.g. via a template).
// CHECK: func.func @debug_arr
// CHECK: emitc.call_opaque "__heir_debug"
// CHECK-SAME: !emitc.opaque<"const std::array<Ciphertext<word>, 1>&">
func.func private @__heir_debug_1(!encoder, !user_interface, tensor<1x!ciphertext>)
func.func @debug_arr(%enc: !encoder, %ui: !user_interface, %ct: tensor<1x!ciphertext>) {
  func.call @__heir_debug_1(%enc, %ui, %ct) {debug.name = "arr0", debug.metadata = "m"} : (!encoder, !user_interface, tensor<1x!ciphertext>) -> ()
  return
}
