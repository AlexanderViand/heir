// Input for the cheddar-to-emitc *compile* test (see BUILD): every function
// here is lowered to C++ and compiled against cheddar_stub.h. The point is to
// exercise the emitter's move/const handling on the op surface that real
// kernels use, with ctx / user_interface / keys / evk_map taken as function
// arguments (the shape a CKKS-to-Cheddar lowering produces).
//
// Destination-passing form: each payload-producing cheddar op takes an explicit
// `bufferization.alloc_tensor` destination (its `outs` init), a scalar payload
// is a rank-0 `tensor<!cheddar.X>`, and the whole module flows through the same
// pipeline the e2e examples use (one-shot-bufferize -> buffer-results-to-out-
// params -> convert-to-emitc -> cheddar-emitc-boundary).
//
// The setup/getter ops (create_context, create_user_interface, get_encoder,
// get_evk_map, get_mult_key, encode/decode) are intentionally absent: that
// part of the emitter has independent, pre-existing API mismatches and is
// tracked separately.

!ciphertext = !cheddar.ciphertext
!plaintext = !cheddar.plaintext
!constant = !cheddar.constant
!context = !cheddar.context

// Add / Sub / Mult chained on ciphertexts.
func.func @arith(%ctx: !context, %a: tensor<!ciphertext>,
                 %b: tensor<!ciphertext>) -> tensor<!ciphertext> {
  %d0 = bufferization.alloc_tensor() : tensor<!ciphertext>
  %0 = cheddar.add %ctx, %a, %b, %d0
      : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d1 = bufferization.alloc_tensor() : tensor<!ciphertext>
  %1 = cheddar.sub %ctx, %0, %b, %d1
      : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d2 = bufferization.alloc_tensor() : tensor<!ciphertext>
  %2 = cheddar.mult %ctx, %1, %a, %d2
      : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %2 : tensor<!ciphertext>
}

// ct+pt and ct+const overloaded dispatch.
func.func @ct_pt_const(%ctx: !context, %ct: tensor<!ciphertext>,
                       %pt: tensor<!plaintext>, %c: tensor<!constant>)
    -> tensor<!ciphertext> {
  %d0 = bufferization.alloc_tensor() : tensor<!ciphertext>
  %0 = cheddar.add_plain %ctx, %ct, %pt, %d0
      : (!context, tensor<!ciphertext>, tensor<!plaintext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d1 = bufferization.alloc_tensor() : tensor<!ciphertext>
  %1 = cheddar.sub_plain %ctx, %0, %pt, %d1
      : (!context, tensor<!ciphertext>, tensor<!plaintext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d2 = bufferization.alloc_tensor() : tensor<!ciphertext>
  %2 = cheddar.mult_plain %ctx, %1, %pt, %d2
      : (!context, tensor<!ciphertext>, tensor<!plaintext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d3 = bufferization.alloc_tensor() : tensor<!ciphertext>
  %3 = cheddar.add_const %ctx, %2, %c, %d3
      : (!context, tensor<!ciphertext>, tensor<!constant>, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d4 = bufferization.alloc_tensor() : tensor<!ciphertext>
  %4 = cheddar.mult_const %ctx, %3, %c, %d4
      : (!context, tensor<!ciphertext>, tensor<!constant>, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %4 : tensor<!ciphertext>
}

// Unary ops.
func.func @unary(%ctx: !context, %ct: tensor<!ciphertext>)
    -> tensor<!ciphertext> {
  %d0 = bufferization.alloc_tensor() : tensor<!ciphertext>
  %0 = cheddar.neg %ctx, %ct, %d0
      : (!context, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d1 = bufferization.alloc_tensor() : tensor<!ciphertext>
  %1 = cheddar.rescale %ctx, %0, %d1
      : (!context, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d2 = bufferization.alloc_tensor() : tensor<!ciphertext>
  %2 = cheddar.level_down %ctx, %1, %d2 {targetLevel = 2 : i64}
      : (!context, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %2 : tensor<!ciphertext>
}

// Relinearize / RelinearizeRescale with an evaluation-key argument.
func.func @relin(%ctx: !context, %ct: tensor<!ciphertext>,
                 %k: !cheddar.eval_key) -> tensor<!ciphertext> {
  %d0 = bufferization.alloc_tensor() : tensor<!ciphertext>
  %0 = cheddar.relinearize %ctx, %ct, %k, %d0
      : (!context, tensor<!ciphertext>, !cheddar.eval_key, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d1 = bufferization.alloc_tensor() : tensor<!ciphertext>
  %1 = cheddar.relinearize_rescale %ctx, %0, %k, %d1
      : (!context, tensor<!ciphertext>, !cheddar.eval_key, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %1 : tensor<!ciphertext>
}

// HMult with an evaluation-key argument.
func.func @hmult(%ctx: !context, %a: tensor<!ciphertext>,
                 %b: tensor<!ciphertext>, %k: !cheddar.eval_key)
    -> tensor<!ciphertext> {
  %d0 = bufferization.alloc_tensor() : tensor<!ciphertext>
  %0 = cheddar.hmult %ctx, %a, %b, %k, %d0 {rescale = true}
      : (!context, tensor<!ciphertext>, tensor<!ciphertext>, !cheddar.eval_key, tensor<!ciphertext>)
      -> tensor<!ciphertext>
  return %0 : tensor<!ciphertext>
}

// Rotation / conjugation: the key is looked up inline via the UserInterface
// argument, so these functions must carry a user_interface arg.
func.func @rotations(%ctx: !context, %ui: !cheddar.user_interface,
                     %a: tensor<!ciphertext>, %b: tensor<!ciphertext>)
    -> tensor<!ciphertext> {
  %d0 = bufferization.alloc_tensor() : tensor<!ciphertext>
  %0 = cheddar.hrot %ctx, %a, %d0 {static_distance = 5 : i64}
      : (!context, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d1 = bufferization.alloc_tensor() : tensor<!ciphertext>
  %1 = cheddar.hrot_add %ctx, %0, %b, %d1 {distance = 7 : i64}
      : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d2 = bufferization.alloc_tensor() : tensor<!ciphertext>
  %2 = cheddar.hconj %ctx, %1, %d2
      : (!context, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d3 = bufferization.alloc_tensor() : tensor<!ciphertext>
  %3 = cheddar.hconj_add %ctx, %2, %b, %d3
      : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %3 : tensor<!ciphertext>
}

// mad_unsafe with a *local* accumulator (the result of add): the accumulator
// is the in-place DPS init, so MadUnsafe(acc, ...) binds fine. This path
// already compiled; it's here as the control case.
func.func @mad_local(%ctx: !context, %a: tensor<!ciphertext>,
                     %b: tensor<!ciphertext>, %c: tensor<!constant>)
    -> tensor<!ciphertext> {
  %d0 = bufferization.alloc_tensor() : tensor<!ciphertext>
  %acc = cheddar.add %ctx, %a, %b, %d0
      : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  %r = cheddar.mad_unsafe %ctx, %acc, %a, %c
      : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!constant>)
      -> tensor<!ciphertext>
  return %r : tensor<!ciphertext>
}

// mad_unsafe with the accumulator coming straight from a *function argument*
// (finding 1). The accumulator is mutated in place by MadUnsafe and then
// returned, so it must be lifted to a mutable `Ct&` -- not the `const Ct&`
// that the by-value-arg tightening would otherwise produce.
func.func @mad_arg(%ctx: !context, %acc: tensor<!ciphertext>,
                   %in: tensor<!ciphertext>, %c: tensor<!constant>)
    -> tensor<!ciphertext> {
  %r = cheddar.mad_unsafe %ctx, %acc, %in, %c
      : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!constant>)
      -> tensor<!ciphertext>
  return %r : tensor<!ciphertext>
}

// Identity: returning a move-only argument unchanged must lift the arg to an
// in-place `Ct&` out-param, not copy.
func.func @identity(%ct: tensor<!ciphertext>) -> tensor<!ciphertext> {
  return %ct : tensor<!ciphertext>
}

// Bootstrapping-family ops taking an EvkMap argument (const EvkMap& at the C++
// boundary).
func.func @boot(%ctx: !cheddar.boot_context, %ct: tensor<!ciphertext>,
                %evk: !cheddar.evk_map) -> tensor<!ciphertext> {
  %d0 = bufferization.alloc_tensor() : tensor<!ciphertext>
  %0 = cheddar.boot %ctx, %ct, %evk, %d0
      : (!cheddar.boot_context, tensor<!ciphertext>, !cheddar.evk_map, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %0 : tensor<!ciphertext>
}

// The diagonals are a float tensor; the emitter renders it as a 2D
// `double[][W]` array passed to RunLinearTransform.
func.func @linear_transform(%ctx: !context, %ct: tensor<!ciphertext>,
                            %evk: !cheddar.evk_map, %d: tensor<2x4xf64>)
    -> tensor<!ciphertext> {
  %d0 = bufferization.alloc_tensor() : tensor<!ciphertext>
  %0 = cheddar.linear_transform %ctx, %ct, %evk, %d, %d0
      {diagonal_indices = array<i32: 0, 1>, level = 5 : i64, bs = 2 : i64, gs = 1 : i64}
      : (!context, tensor<!ciphertext>, !cheddar.evk_map, tensor<2x4xf64>, tensor<!ciphertext>)
      -> tensor<!ciphertext>
  return %0 : tensor<!ciphertext>
}

func.func @eval_poly(%ctx: !context, %ct: tensor<!ciphertext>,
                     %evk: !cheddar.evk_map) -> tensor<!ciphertext> {
  %d0 = bufferization.alloc_tensor() : tensor<!ciphertext>
  %0 = cheddar.eval_poly %ctx, %ct, %evk, %d0
      {coefficients = [1.0 : f64, 2.0 : f64, 3.0 : f64], level = 4 : i64, outputLevel = 3 : i64}
      : (!context, tensor<!ciphertext>, !cheddar.evk_map, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %0 : tensor<!ciphertext>
}

// Encrypt / Decrypt out-param calls on the UserInterface.
func.func @encrypt_decrypt(%ui: !cheddar.user_interface, %pt: tensor<!plaintext>,
                           %ct: tensor<!ciphertext>)
    -> (tensor<!ciphertext>, tensor<!plaintext>) {
  %d0 = bufferization.alloc_tensor() : tensor<!ciphertext>
  %0 = cheddar.encrypt %ui, %pt, %d0
      : (!cheddar.user_interface, tensor<!plaintext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d1 = bufferization.alloc_tensor() : tensor<!plaintext>
  %1 = cheddar.decrypt %ui, %ct, %d1
      : (!cheddar.user_interface, tensor<!ciphertext>, tensor<!plaintext>) -> tensor<!plaintext>
  return %0, %1 : tensor<!ciphertext>, tensor<!plaintext>
}

// Destination-passing loop kernel: an scf.for that writes a move-only payload
// into element `i` of the output via tensor.insert_slice. After bufferization
// this is a loop over a `memref<8x!cheddar.ciphertext>`; the output boundary
// type must be a mutable `std::array<Ciphertext<word>, 8>&` (a `const` one
// would reject the per-iteration `arr[i] = std::move(...)` store).
func.func @loop_store(%ctx: !context, %in: tensor<!ciphertext>,
                      %out: tensor<8x!ciphertext>) -> tensor<8x!ciphertext> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %r = scf.for %i = %c0 to %c8 step %c1 iter_args(%acc = %out)
      -> (tensor<8x!ciphertext>) {
    %d = bufferization.alloc_tensor() : tensor<!ciphertext>
    %v = cheddar.add %ctx, %in, %in, %d
        : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>)
        -> tensor<!ciphertext>
    %ins = tensor.insert_slice %v into %acc[%i] [1] [1]
        : tensor<!ciphertext> into tensor<8x!ciphertext>
    scf.yield %ins : tensor<8x!ciphertext>
  }
  return %r : tensor<8x!ciphertext>
}
