// RUN: heir-opt --cheddar-bufferize --fold-memref-alias-ops --cse --canonicalize --convert-to-emitc=filter-dialects=cheddar,arith,scf --cheddar-emitc-boundary --reconcile-unrealized-casts %s | FileCheck %s

// Function-boundary handling for the move-only CHEDDAR payload/handle types.
// `--cheddar-bufferize` exposes each payload result as a trailing out-param
// before One-Shot, and `--cheddar-emitc-boundary` re-types the args: a read-only
// payload buffer arg is `const Ciphertext<word>&`, a written one (an out-param
// or an in-place accumulator) is a mutable `Ciphertext<word>&`, and the
// non-copyable handle types (Encoder / EvkMap / EvaluationKey) are `const T&`.

!ciphertext = !cheddar.ciphertext
!constant = !cheddar.constant
!context = !cheddar.context
!evk_map = !cheddar.evk_map
!boot_context = !cheddar.boot_context

// An EvkMap argument is a non-copyable handle, so it tightens to
// `const EvkMap<word>&` rather than staying a by-value parameter.
// CHECK: func.func @boot(
// CHECK-SAME: !emitc.opaque<"const EvkMap<word>&">
func.func @boot(%ctx: !boot_context, %ct: tensor<!ciphertext>, %evk: !evk_map)
    -> tensor<!ciphertext> {
  %d = tensor.empty() : tensor<!ciphertext>
  %0 = cheddar.boot %ctx, %ct, %evk, %d
      : (!boot_context, tensor<!ciphertext>, !evk_map, tensor<!ciphertext>)
      -> tensor<!ciphertext>
  return %0 : tensor<!ciphertext>
}
