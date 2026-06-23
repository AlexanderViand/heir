// RUN: heir-opt "--one-shot-bufferize=bufferize-function-boundaries=true function-boundary-type-conversion=identity-layout-map" "--buffer-results-to-out-params=hoist-static-allocs=true modify-public-functions=true add-result-attr=true" --fold-memref-alias-ops --canonicalize --convert-to-emitc --cheddar-emitc-boundary --reconcile-unrealized-casts %s | FileCheck %s

// End-to-end op coverage for the cheddar -> EmitC lowering, now driven by stock
// `--convert-to-emitc` (cheddar's dialect interface) + the
// `--cheddar-emitc-boundary` pass, over destination-passing-style cheddar ops.
// Every payload-producing op carries a `bufferization.alloc_tensor` `$output`
// destination; bufferization + `--buffer-results-to-out-params` turn func
// results into trailing out-params, and the boundary pass re-types move-only
// payload args as C++ references (`const T&` inputs, `T&` out-params).

!ciphertext = !cheddar.ciphertext
!plaintext = !cheddar.plaintext
!constant = !cheddar.constant
!context = !cheddar.context
!encoder = !cheddar.encoder
!eval_key = !cheddar.eval_key
!evk_map = !cheddar.evk_map
!user_interface = !cheddar.user_interface
!parameter = !cheddar.parameter
!boot_context = !cheddar.boot_context

// CreateContext is a static factory (not destination-passing): it produces a
// `Context<word>*`, so it stays `T x = T::Create(args);`.
// CHECK: func.func @create_context
// CHECK: emitc.call_opaque "Context<word>::Create"
func.func @create_context(%params: !parameter) -> !context {
  %ctx = cheddar.create_context %params : (!parameter) -> !context
  return %ctx : !context
}

// CHECK: func.func @prepare_keys
// CHECK: emitc.verbatim "{}->PrepareRotationKey(3, 5);"
func.func @prepare_keys(%ui: !user_interface) {
  cheddar.prepare_rot_key %ui {distance = 3 : i64, maxLevel = 5 : i64} : (!user_interface) -> ()
  return
}

// A two-op chain: each op is `ctx->Method(out, a, b)`. The first op's result is
// an intermediate local (`emitc.variable`); the second writes the function
// out-param. Inputs are `const Ciphertext<word>&`, the out-param is mutable.
// CHECK: func.func @arith(
// CHECK-SAME: !emitc.ptr<!emitc.opaque<"Context<word>">>
// CHECK-SAME: !emitc.opaque<"const Ciphertext<word>&">
// CHECK-SAME: !emitc.opaque<"Ciphertext<word>&">
// CHECK: emitc.member_call_opaque %arg0 "Add"(%[[V:.*]], %arg1, %arg2)
// CHECK: emitc.member_call_opaque %arg0 "Mult"(%arg3, %[[V]], %arg2)
func.func @arith(%ctx: !context, %a: tensor<!ciphertext>, %b: tensor<!ciphertext>) -> tensor<!ciphertext> {
  %d0 = bufferization.alloc_tensor() : tensor<!ciphertext>
  %r = cheddar.add %ctx, %a, %b, %d0 : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d1 = bufferization.alloc_tensor() : tensor<!ciphertext>
  %s = cheddar.mult %ctx, %r, %b, %d1 : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %s : tensor<!ciphertext>
}

// CHEDDAR's Context overloads Add/Sub/Mult on the second operand type, so the
// `*_plain` / `*_const` ops dispatch to the base method name.
// CHECK: func.func @plain_const
// CHECK: emitc.member_call_opaque %arg0 "Add"
// CHECK: emitc.member_call_opaque %arg0 "Mult"
func.func @plain_const(%ctx: !context, %ct: tensor<!ciphertext>, %pt: tensor<!plaintext>, %c: tensor<!constant>) -> tensor<!ciphertext> {
  %d0 = bufferization.alloc_tensor() : tensor<!ciphertext>
  %r1 = cheddar.add_plain %ctx, %ct, %pt, %d0 : (!context, tensor<!ciphertext>, tensor<!plaintext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d1 = bufferization.alloc_tensor() : tensor<!ciphertext>
  %r2 = cheddar.mult_const %ctx, %r1, %c, %d1 : (!context, tensor<!ciphertext>, tensor<!constant>, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %r2 : tensor<!ciphertext>
}

// The level_down target level is appended as a trailing opaque constant arg.
// CHECK: func.func @unary
// CHECK: emitc.member_call_opaque %arg0 "Neg"
// CHECK: emitc.member_call_opaque %arg0 "LevelDown"
// CHECK-SAME: #emitc.opaque<"2">
func.func @unary(%ctx: !context, %ct: tensor<!ciphertext>) -> tensor<!ciphertext> {
  %d0 = bufferization.alloc_tensor() : tensor<!ciphertext>
  %n = cheddar.neg %ctx, %ct, %d0 : (!context, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d1 = bufferization.alloc_tensor() : tensor<!ciphertext>
  %l = cheddar.level_down %ctx, %n, %d1 {targetLevel = 2 : i64} : (!context, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %l : tensor<!ciphertext>
}

// CHECK: func.func @relin
// CHECK: emitc.member_call_opaque %arg0 "Relinearize"
// CHECK: emitc.member_call_opaque %arg0 "Rescale"
func.func @relin(%ctx: !context, %ct: tensor<!ciphertext>, %k: !eval_key) -> tensor<!ciphertext> {
  %d0 = bufferization.alloc_tensor() : tensor<!ciphertext>
  %r1 = cheddar.relinearize %ctx, %ct, %k, %d0 : (!context, tensor<!ciphertext>, !eval_key, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d1 = bufferization.alloc_tensor() : tensor<!ciphertext>
  %r2 = cheddar.rescale %ctx, %r1, %d1 : (!context, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %r2 : tensor<!ciphertext>
}

// The HMult `rescale` flag is appended as the trailing opaque constant arg.
// CHECK: func.func @hmult
// CHECK: emitc.member_call_opaque %arg0 "HMult"
// CHECK-SAME: #emitc.opaque<"true">
func.func @hmult(%ctx: !context, %a: tensor<!ciphertext>, %b: tensor<!ciphertext>, %k: !eval_key) -> tensor<!ciphertext> {
  %d0 = bufferization.alloc_tensor() : tensor<!ciphertext>
  %r = cheddar.hmult %ctx, %a, %b, %k, %d0 {rescale = true} : (!context, tensor<!ciphertext>, tensor<!ciphertext>, !eval_key, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %r : tensor<!ciphertext>
}

// MadUnsafe is in-place on its accumulator. Here the accumulator is an input
// arg returned as the result, so it is mutable and the out-param is filled by a
// move from it.
// CHECK: func.func @mad
// CHECK: emitc.member_call_opaque %arg0 "MadUnsafe"(%arg1, %arg2, %arg3)
// CHECK: emitc.verbatim "{} = std::move({});"
func.func @mad(%ctx: !context, %acc: tensor<!ciphertext>, %in: tensor<!ciphertext>, %c: tensor<!constant>) -> tensor<!ciphertext> {
  %r = cheddar.mad_unsafe %ctx, %acc, %in, %c : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!constant>) -> tensor<!ciphertext>
  return %r : tensor<!ciphertext>
}

// Encode bridges a float message buffer through a std::vector<Complex> (stays
// verbatim, using the encoder's canonical per-level scale); encrypt is an
// out-param method call.
// CHECK: func.func @enc_chain
// CHECK: emitc.verbatim "{}.Encode({}, 5, {}.GetScale(5), {});"
// CHECK: emitc.member_call_opaque %arg2 "Encrypt"
func.func @enc_chain(%enc: !encoder, %msg: tensor<4xf64>, %ui: !user_interface) -> tensor<!ciphertext> {
  %dp = bufferization.alloc_tensor() : tensor<!plaintext>
  %pt = cheddar.encode %enc, %msg, %dp {level = 5 : i64, scale = 68719476736.0 : f64} : (!encoder, tensor<4xf64>, tensor<!plaintext>) -> tensor<!plaintext>
  %dc = bufferization.alloc_tensor() : tensor<!ciphertext>
  %ct = cheddar.encrypt %ui, %pt, %dc : (!user_interface, tensor<!plaintext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %ct : tensor<!ciphertext>
}

// Decrypt is an out-param method call; decode reads into a temporary
// std::vector<Complex> then copies the real parts into the float buffer.
// CHECK: func.func @dec_chain
// CHECK: emitc.member_call_opaque %arg1 "Decrypt"
// CHECK: emitc.verbatim "{}.Decode({}, {});"
func.func @dec_chain(%enc: !encoder, %ui: !user_interface, %ct: tensor<!ciphertext>, %dst: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %dp = bufferization.alloc_tensor() : tensor<!plaintext>
  %pt = cheddar.decrypt %ui, %ct, %dp : (!user_interface, tensor<!ciphertext>, tensor<!plaintext>) -> tensor<!plaintext>
  %msg = cheddar.decode %enc, %pt, %dst : (!encoder, tensor<!plaintext>, tensor<1x4xf32>) -> tensor<1x4xf32>
  return %msg : tensor<1x4xf32>
}

// HRot/HConj keep their verbatim form (the rotation/conjugation key is a nested
// `ui->GetRotationKey(d)` lookup). Static distance bakes the distance into the
// format string; dynamic distance threads the SSA value twice.
// CHECK: func.func @hrot_static
// CHECK: emitc.verbatim "{}->HRot({}, {}, {}->GetRotationKey(5), 5);"
func.func @hrot_static(%ctx: !context, %ui: !user_interface, %ct: tensor<!ciphertext>) -> tensor<!ciphertext> {
  %d0 = bufferization.alloc_tensor() : tensor<!ciphertext>
  %r = cheddar.hrot %ctx, %ct, %d0 {static_distance = 5 : i64} : (!context, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %r : tensor<!ciphertext>
}

// CHECK: func.func @hrot_dyn
// CHECK: emitc.verbatim "{}->HRot({}, {}, {}->GetRotationKey({}), {});"
// CHECK-SAME: %arg3, %arg3
func.func @hrot_dyn(%ctx: !context, %ui: !user_interface, %ct: tensor<!ciphertext>, %d: index) -> tensor<!ciphertext> {
  %d0 = bufferization.alloc_tensor() : tensor<!ciphertext>
  %r = cheddar.hrot %ctx, %ct, %d0, %d : (!context, tensor<!ciphertext>, tensor<!ciphertext>, index) -> tensor<!ciphertext>
  return %r : tensor<!ciphertext>
}

// CHECK: func.func @hconj_add
// CHECK: emitc.verbatim "{}->HConjAdd({}, {}, {}, {}->GetConjugationKey());"
func.func @hconj_add(%ctx: !context, %ui: !user_interface, %a: tensor<!ciphertext>, %b: tensor<!ciphertext>) -> tensor<!ciphertext> {
  %d0 = bufferization.alloc_tensor() : tensor<!ciphertext>
  %r = cheddar.hconj_add %ctx, %a, %b, %d0 : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %r : tensor<!ciphertext>
}

// Boot is a BootContext method; cheddar.boot requires a !cheddar.boot_context
// (lowered to BootContext<word>*) so no downcast is needed.
// CHECK: func.func @boot
// CHECK: emitc.member_call_opaque %arg0 "Boot"
func.func @boot(%ctx: !boot_context, %ct: tensor<!ciphertext>, %evk: !evk_map) -> tensor<!ciphertext> {
  %d0 = bufferization.alloc_tensor() : tensor<!ciphertext>
  %r = cheddar.boot %ctx, %ct, %evk, %d0 : (!boot_context, tensor<!ciphertext>, !evk_map, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %r : tensor<!ciphertext>
}

// linear_transform / eval_poly have no Context method (they are classes), so
// they lower to one structured call to the HEIR-side RunLinearTransform /
// RunEvalPoly shim, carrying the trailing literal args + template args.
// CHECK: func.func @lintrans
// CHECK: emitc.call_opaque "RunLinearTransform"
// CHECK-SAME: 0, 1}, 5, 2, 1
// CHECK-SAME: word
func.func @lintrans(%ctx: !context, %ct: tensor<!ciphertext>, %evk: !evk_map, %dg: tensor<2x4xf64>) -> tensor<!ciphertext> {
  %d0 = bufferization.alloc_tensor() : tensor<!ciphertext>
  %r = cheddar.linear_transform %ctx, %ct, %evk, %dg, %d0 {diagonal_indices = array<i32: 0, 1>, level = 5 : i64, bs = 2 : i64, gs = 1 : i64} : (!context, tensor<!ciphertext>, !evk_map, tensor<2x4xf64>, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %r : tensor<!ciphertext>
}

// CHECK: func.func @eval_poly
// CHECK: emitc.call_opaque "RunEvalPoly"
// CHECK-SAME: 2, 3}, 4, 3
// CHECK-SAME: word
func.func @eval_poly(%ctx: !context, %ct: tensor<!ciphertext>, %evk: !evk_map) -> tensor<!ciphertext> {
  %d0 = bufferization.alloc_tensor() : tensor<!ciphertext>
  %r = cheddar.eval_poly %ctx, %ct, %evk, %d0 {coefficients = [1.0 : f64, 2.0 : f64, 3.0 : f64], level = 4 : i64, outputLevel = 3 : i64} : (!context, tensor<!ciphertext>, !evk_map, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %r : tensor<!ciphertext>
}
