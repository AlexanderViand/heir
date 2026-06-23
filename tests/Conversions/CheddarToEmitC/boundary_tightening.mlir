// RUN: heir-opt "--one-shot-bufferize=bufferize-function-boundaries=true function-boundary-type-conversion=identity-layout-map" "--buffer-results-to-out-params=hoist-static-allocs=true modify-public-functions=true add-result-attr=true" --fold-memref-alias-ops --canonicalize --convert-to-emitc --cheddar-emitc-boundary --reconcile-unrealized-casts %s | FileCheck %s

// Function-boundary handling for the move-only CHEDDAR payload/handle types.
// `--buffer-results-to-out-params` lifts each payload result to a trailing
// out-param, and `--cheddar-emitc-boundary` re-types the args: a read-only
// payload buffer arg is `const Ciphertext<word>&`, a written one (an out-param
// or an in-place accumulator) is a mutable `Ciphertext<word>&`, and the
// non-copyable handle types (Encoder / EvkMap / EvaluationKey) are `const T&`.

!ciphertext = !cheddar.ciphertext
!constant = !cheddar.constant
!context = !cheddar.context
!evk_map = !cheddar.evk_map
!boot_context = !cheddar.boot_context

// Returning an input ciphertext unchanged: the result becomes a separate
// out-param filled by a `std::move` (a C++ copy is illegal for the move-only
// payload). The moved-from input must therefore be a *mutable* `Ciphertext&` --
// `std::move` cannot bind a `const T&` -- so both args are mutable refs.
// CHECK: func.func @identity(
// CHECK-SAME: !emitc.opaque<"Ciphertext<word>&">
// CHECK-SAME: !emitc.opaque<"Ciphertext<word>&">
// CHECK: emitc.verbatim "{} = std::move({});"
func.func @identity(%ct: tensor<!ciphertext>) -> tensor<!ciphertext> {
  return %ct : tensor<!ciphertext>
}

// MadUnsafe mutates its accumulator in place: the accumulator arg is a mutable
// `Ciphertext<word>&` (it is written), the other inputs are `const ...&`, and
// the returned value is moved into the out-param.
// CHECK: func.func @mad_arg(
// CHECK-SAME: !emitc.opaque<"Ciphertext<word>&">
// CHECK-SAME: !emitc.opaque<"const Ciphertext<word>&">
// CHECK-SAME: !emitc.opaque<"const Constant<word>&">
// CHECK-SAME: !emitc.opaque<"Ciphertext<word>&">
// CHECK: emitc.member_call_opaque %arg0 "MadUnsafe"(%arg1, %arg2, %arg3)
// CHECK: emitc.verbatim "{} = std::move({});"
func.func @mad_arg(%ctx: !context, %acc: tensor<!ciphertext>,
                   %in: tensor<!ciphertext>, %c: tensor<!constant>)
    -> tensor<!ciphertext> {
  %r = cheddar.mad_unsafe %ctx, %acc, %in, %c
      : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!constant>)
      -> tensor<!ciphertext>
  return %r : tensor<!ciphertext>
}

// An EvkMap argument is a non-copyable handle, so it tightens to
// `const EvkMap<word>&` rather than staying a by-value parameter.
// CHECK: func.func @boot(
// CHECK-SAME: !emitc.opaque<"const EvkMap<word>&">
func.func @boot(%ctx: !boot_context, %ct: tensor<!ciphertext>, %evk: !evk_map)
    -> tensor<!ciphertext> {
  %d = bufferization.alloc_tensor() : tensor<!ciphertext>
  %0 = cheddar.boot %ctx, %ct, %evk, %d
      : (!boot_context, tensor<!ciphertext>, !evk_map, tensor<!ciphertext>)
      -> tensor<!ciphertext>
  return %0 : tensor<!ciphertext>
}
