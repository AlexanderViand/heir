// RUN: heir-opt --cheddar-bufferize --fold-memref-alias-ops --cse --canonicalize --drop-equivalent-buffer-results "--buffer-results-to-out-params=hoist-static-allocs=true modify-public-functions=true add-result-attr=true" --canonicalize --convert-to-emitc=filter-dialects=cheddar,arith,scf --cheddar-emitc-boundary --reconcile-unrealized-casts %s | FileCheck %s

// End-to-end op coverage for the cheddar -> EmitC lowering, now driven by stock
// `--convert-to-emitc` (cheddar's dialect interface) + the
// `--cheddar-emitc-boundary` pass, over destination-passing-style cheddar ops.
// Every payload-producing op carries a `tensor.empty` `$output`
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

// CHECK: emitc.global static @resource : !emitc.array<4xf32> = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]>
memref.global "private" constant @resource : memref<4xf32> = dense_resource<weights>

// C++ zero-initializes private globals with no initializer. Dropping a
// positive-zero splat keeps large generated arrays compact.
// CHECK: emitc.global static @zero_splat : !emitc.array<4x4xf32>{{$}}
memref.global "private" constant @zero_splat : memref<4x4xf32> = dense<0.000000e+00>

// Negative floating-point zero must remain explicit.
// CHECK: emitc.global static @negative_zero_splat : !emitc.array<4xf32> = dense<-0.000000e+00>
memref.global "private" constant @negative_zero_splat : memref<4xf32> = dense<-0.000000e+00>

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
  %d0 = tensor.empty() : tensor<!ciphertext>
  %r = cheddar.add %ctx, %a, %b, %d0 : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d1 = tensor.empty() : tensor<!ciphertext>
  %s = cheddar.mult %ctx, %r, %b, %d1 : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %s : tensor<!ciphertext>
}

// A semantic ciphertext copy lowers to CHEDDAR's deep-copy API, never C++
// copy-assignment on the move-only payload.
// CHECK: func.func @copy
// CHECK: emitc.member_call_opaque %arg0 "Copy"(%arg2, %arg1)
func.func @copy(%ctx: !context, %input: tensor<!ciphertext>) -> tensor<!ciphertext> {
  %dest = tensor.empty() : tensor<!ciphertext>
  %result = cheddar.copy %ctx, %input, %dest : (!context, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %result : tensor<!ciphertext>
}

// A memref.copy that survives alias folding has true copy semantics and lowers
// through the same CHEDDAR deep-copy API.
// CHECK: func.func @memref_copy
// CHECK: emitc.member_call_opaque %arg0 "Copy"(%arg2, %arg1)
func.func @memref_copy(%ctx: !context, %input: memref<!ciphertext>,
                       %output: memref<!ciphertext>) {
  memref.copy %input, %output
      : memref<!ciphertext> to memref<!ciphertext>
  return
}

// CHEDDAR's Context overloads Add/Sub/Mult on the second operand type, so the
// `*_plain` / `*_const` ops dispatch to the base method name.
// CHECK: func.func @plain_const
// CHECK: emitc.member_call_opaque %arg0 "Add"
// CHECK: emitc.member_call_opaque %arg0 "Mult"
func.func @plain_const(%ctx: !context, %ct: tensor<!ciphertext>, %pt: tensor<!plaintext>, %c: tensor<!constant>) -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %r1 = cheddar.add_plain %ctx, %ct, %pt, %d0 : (!context, tensor<!ciphertext>, tensor<!plaintext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d1 = tensor.empty() : tensor<!ciphertext>
  %r2 = cheddar.mult_const %ctx, %r1, %c, %d1 : (!context, tensor<!ciphertext>, tensor<!constant>, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %r2 : tensor<!ciphertext>
}

// The level_down target level is appended as a trailing opaque constant arg.
// CHECK: func.func @unary
// CHECK: emitc.member_call_opaque %arg0 "Neg"
// CHECK: emitc.member_call_opaque %arg0 "LevelDown"
// CHECK-SAME: #emitc.opaque<"2">
func.func @unary(%ctx: !context, %ct: tensor<!ciphertext>) -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %n = cheddar.neg %ctx, %ct, %d0 : (!context, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d1 = tensor.empty() : tensor<!ciphertext>
  %l = cheddar.level_down %ctx, %n, %d1 {targetLevel = 2 : i64} : (!context, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %l : tensor<!ciphertext>
}

// CHECK: func.func @relin
// CHECK: emitc.member_call_opaque %arg0 "Relinearize"
// CHECK: emitc.member_call_opaque %arg0 "Rescale"
func.func @relin(%ctx: !context, %ct: tensor<!ciphertext>, %k: !eval_key) -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %r1 = cheddar.relinearize %ctx, %ct, %k, %d0 : (!context, tensor<!ciphertext>, !eval_key, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d1 = tensor.empty() : tensor<!ciphertext>
  %r2 = cheddar.rescale %ctx, %r1, %d1 : (!context, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %r2 : tensor<!ciphertext>
}

// The HMult `rescale` flag is appended as the trailing opaque constant arg.
// CHECK: func.func @hmult
// CHECK: emitc.member_call_opaque %arg0 "HMult"
// CHECK-SAME: #emitc.opaque<"true">
func.func @hmult(%ctx: !context, %a: tensor<!ciphertext>, %b: tensor<!ciphertext>, %k: !eval_key) -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %r = cheddar.hmult %ctx, %a, %b, %k, %d0 {rescale = true} : (!context, tensor<!ciphertext>, tensor<!ciphertext>, !eval_key, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %r : tensor<!ciphertext>
}

// Encode bridges a float message buffer through a std::vector<Complex> and
// uses CHEDDAR's canonical per-level scale; encrypt is an out-param method
// call.
// CHECK: func.func @enc_chain
// CHECK-SAME: !emitc.ptr<f64>
// CHECK: emitc.verbatim "{}.Encode({}, 5, {}.GetScale(5), {});"
// CHECK: emitc.member_call_opaque %arg2 "Encrypt"
func.func @enc_chain(%enc: !encoder, %msg: tensor<4xf64>, %ui: !user_interface) -> tensor<!ciphertext> {
  %dp = tensor.empty() : tensor<!plaintext>
  %pt = cheddar.encode %enc, %msg, %dp {level = 5 : i64, logScale = 37 : i64} : (!encoder, tensor<4xf64>, tensor<!plaintext>) -> tensor<!plaintext>
  %dc = tensor.empty() : tensor<!ciphertext>
  %ct = cheddar.encrypt %ui, %pt, %dc : (!user_interface, tensor<!plaintext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %ct : tensor<!ciphertext>
}

// Decrypt is an out-param method call; decode reads into a temporary
// std::vector<Complex> then copies the real parts into the float buffer.
// CHECK: func.func @dec_chain
// CHECK-SAME: !emitc.ptr<f32>
// CHECK: emitc.member_call_opaque %arg1 "Decrypt"
// CHECK: emitc.verbatim "{}.Decode({}, {});"
// CHECK: emitc.verbatim "for (size_t _i = 0; _i < 4; ++_i) {}[_i] = {}.at(_i).real();"
func.func @dec_chain(%enc: !encoder, %ui: !user_interface, %ct: tensor<!ciphertext>, %dst: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %dp = tensor.empty() : tensor<!plaintext>
  %pt = cheddar.decrypt %ui, %ct, %dp : (!user_interface, tensor<!ciphertext>, tensor<!plaintext>) -> tensor<!plaintext>
  %msg = cheddar.decode %enc, %pt, %dst : (!encoder, tensor<!plaintext>, tensor<1x4xf32>) -> tensor<1x4xf32>
  return %msg : tensor<1x4xf32>
}

// HRot/HConj keep their verbatim form. Static distance bakes the distance into
// the format string; dynamic distance threads the SSA value twice.
// CHECK: func.func @hrot_static
// CHECK: emitc.verbatim "{}->HRot({}, {}, {}->GetRotationKey(5), 5);"
func.func @hrot_static(%ctx: !context, %ui: !user_interface, %ct: tensor<!ciphertext>) -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %r = cheddar.hrot %ctx, %ui, %ct, %d0 {static_distance = 5 : i64} : (!context, !user_interface, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %r : tensor<!ciphertext>
}

// CHECK: func.func @hrot_dyn
// CHECK: emitc.verbatim "{}->HRot({}, {}, {}->GetRotationKey({}), {});"
// CHECK-SAME: %arg3, %arg3
func.func @hrot_dyn(%ctx: !context, %ui: !user_interface, %ct: tensor<!ciphertext>, %d: index) -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %r = cheddar.hrot %ctx, %ui, %ct, %d0, %d : (!context, !user_interface, tensor<!ciphertext>, tensor<!ciphertext>, index) -> tensor<!ciphertext>
  return %r : tensor<!ciphertext>
}

// CHECK: func.func @hconj_add
// CHECK: emitc.verbatim "{}->HConjAdd({}, {}, {}, {}->GetConjugationKey());"
func.func @hconj_add(%ctx: !context, %ui: !user_interface, %a: tensor<!ciphertext>, %b: tensor<!ciphertext>) -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %r = cheddar.hconj_add %ctx, %ui, %a, %b, %d0 : (!context, !user_interface, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %r : tensor<!ciphertext>
}

// Boot is a BootContext method; cheddar.boot requires a !cheddar.boot_context
// (lowered to BootContext<word>*) so no downcast is needed.
// CHECK: func.func @boot
// CHECK: emitc.member_call_opaque %arg0 "Boot"
func.func @boot(%ctx: !boot_context, %ct: tensor<!ciphertext>, %evk: !evk_map) -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %r = cheddar.boot %ctx, %ct, %evk, %d0 : (!boot_context, tensor<!ciphertext>, !evk_map, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %r : tensor<!ciphertext>
}

// eval_poly lowers to the real EvalPoly<word> class -- there is no
// `RunEvalPoly` in cheddar. It mirrors cheddar's own EvalMod: the level/scale
// are taken from the actual input ciphertext (NPToLevel(in.GetNP()),
// in.GetScale()) and target_scale is the square/divide recurrence over
// GetRescalePrimeProd -- seeding the constructor with anything else silently
// blows the Chebyshev basis recurrence up. The construct/Compile/Evaluate
// (which need the move-only ciphertext as a method receiver + ctx->param_ + the
// recurrence) are emitted as verbatim real-cheddar statements in a `{ }` block
// scope so the EvalPoly (and its GPU power basis) is destroyed right after use.
// CHECK: func.func @eval_poly
// CHECK-SAME: !emitc.opaque<"const Ciphertext<word>&">
// CHECK: emitc.verbatim "{"
// CHECK: emitc.verbatim "ConstContextPtr<word> _ep_cp(ConstContextPtr<word>(), {}
// CHECK: emitc.verbatim "int _ep_lvl = {}->param_.NPToLevel({}.GetNP());"
// CHECK: emitc.verbatim "double _ep_is = {}.GetScale();"
// CHECK: emitc.verbatim "_ep_ts = _ep_ts * _ep_ts / {}->param_.GetRescalePrimeProd
// CHECK: emitc.verbatim "EvalPoly<word> _ep({1, 2, 3}, _ep_lvl, _ep_is, _ep_ts, true);"
// CHECK: emitc.verbatim "_ep.Compile(_ep_cp);"
// CHECK: emitc.verbatim "_ep.Evaluate(_ep_cp, {}, {}, {}.GetMultiplicationKey());"
// CHECK: emitc.verbatim "}"
func.func @eval_poly(%ctx: !context, %enc: !encoder, %ct: tensor<!ciphertext>, %evk: !evk_map) -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %r = cheddar.eval_poly %ctx, %ct, %evk, %d0 {coefficients = [1.0 : f64, 2.0 : f64, 3.0 : f64], levelConsumption = 2 : i64} : (!context, tensor<!ciphertext>, !evk_map, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %r : tensor<!ciphertext>
}

{-#
  dialect_resources: {
    builtin: {
      weights: "0x040000000000803f000000400000404000008040"
    }
  }
#-}
