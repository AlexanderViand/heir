// RUN: heir-opt "--one-shot-bufferize=bufferize-function-boundaries=true function-boundary-type-conversion=identity-layout-map" "--buffer-results-to-out-params=hoist-static-allocs=true modify-public-functions=true add-result-attr=true" --fold-memref-alias-ops --canonicalize --convert-to-emitc --cheddar-emitc-boundary --reconcile-unrealized-casts %s | FileCheck %s

// A destination-passing loop kernel: an scf.for whose body computes a
// ciphertext and writes it into element `i` of the output via
// `tensor.insert_slice`. After bufferization this is a loop over a
// `memref<8x!cheddar.ciphertext>`; the insert_slice becomes a *dynamic-offset*
// rank-reducing subview, which the cheddar EmitC lowering turns into a dynamic
// `emitc.subscript out[i]`. SCF/Arith are lowered to EmitC by their own
// `--convert-to-emitc` interfaces, so the whole loop comes through in one pass:
// the index becomes `emitc.size_t`, the move-only payload buffer boundary is a
// mutable `std::array<...>&`, and the per-iteration write is a subscript +
// `std::move`.

!ciphertext = !cheddar.ciphertext
!context = !cheddar.context

// The ops inside the `emitc.for` body print without the `emitc.` prefix (emitc
// is the body region's default dialect), so match the bare op names there.
// CHECK: func.func @loop_store
// CHECK-SAME: !emitc.opaque<"std::array<Ciphertext<word>, 8>&">
// CHECK: emitc.for
// CHECK: member_call_opaque %arg0 "Add"
// CHECK: subscript %arg2[%arg4]
// CHECK: verbatim "{} = std::move({});"
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
