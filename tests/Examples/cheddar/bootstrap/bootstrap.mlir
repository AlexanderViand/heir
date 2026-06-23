// A real CHEDDAR GPU bootstrap at the cheddar level: encrypt a secret 1024-vector
// at level 0 (the bottom), refresh its noise budget with cheddar.boot, then
// decrypt -- so the result is (approximately) the input. Full-module e2e input
// (client encrypt/decrypt helpers + the boot compute).
//
// This IR is curated rather than pipeline-generated, mirroring the lattigo
// bootstrap example (tests/Examples/lattigo/ckks/bootstrapping): HEIR's CKKS
// param generation produces only a shallow modulus chain, but a real CHEDDAR
// Boot needs a deep bootstrap parameter set (CoeffToSlot + EvalMod + SlotToCoeff
// consume many levels). The harness therefore drives CHEDDAR's curated
// bootparam_40_64bit param set (logN=16, 26-prime chain, scale 2^40,
// num_cts_levels=4, num_stc_levels=3) and this IR only carries the runtime-
// relevant literals: the level-0 encode at scale GetScale(0) = 2^40 (clean at the
// bottom, so no scale bake needed) and the boot/encrypt/decrypt ops. The boot's
// evk_map arg is threaded by --scheme-to-cheddar (LWEToCheddar) and supplied by
// the harness from ui->GetEvkMap() after AddRequiredRotations.
//
// cheddar.boot takes a !cheddar.boot_context and lowers to `ctx->Boot(res, in,
// evk_map)` on a BootContext<word>*; the harness passes the BootContext it built.
!ciphertext = !cheddar.ciphertext
!context = !cheddar.context
!boot_context = !cheddar.boot_context
!encoder = !cheddar.encoder
!eval_key = !cheddar.eval_key
!evk_map = !cheddar.evk_map
!plaintext = !cheddar.plaintext
!user_interface = !cheddar.user_interface
#layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 1024 = 0 and 0 <= i0 <= 1023 and 0 <= slot <= 1023 }">
#original_type = #tensor_ext.original_type<originalType = tensor<1024xf32>, layout = #layout>
module attributes {backend.cheddar, cheddar.logDefaultScale = 40 : i64, cheddar.logN = 16 : i64} {
  func.func @boot(%ctx: !boot_context, %encoder: !encoder, %ui: !user_interface, %evk: !eval_key, %evk_map: !evk_map, %arg0: tensor<1x!ciphertext> {tensor_ext.original_type = #original_type}) -> (tensor<1x!ciphertext> {tensor_ext.original_type = #original_type}) {
    %c0 = arith.constant 0 : index
    %extracted = tensor.extract_slice %arg0[0] [1] [1] : tensor<1x!ciphertext> to tensor<!ciphertext>
    %dps_1 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct = cheddar.boot %ctx, %extracted, %evk_map, %dps_1 : (!boot_context, tensor<!ciphertext>, !evk_map, tensor<!ciphertext>) -> tensor<!ciphertext>
    %0 = tensor.empty() : tensor<1x!ciphertext>
    %inserted = tensor.insert_slice %ct into %0[0] [1] [1] : tensor<!ciphertext> into tensor<1x!ciphertext>
    return %inserted : tensor<1x!ciphertext>
  }
  func.func @boot__encrypt__arg0(%ctx: !context, %encoder: !encoder, %ui: !user_interface, %evk: !eval_key, %arg0: tensor<1024xf32>, %ui_0: !user_interface) -> tensor<1x!ciphertext> attributes {client.enc_func = {func_name = "boot", index = 0 : i64}} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %0 = scf.for %arg1 = %c0_i32 to %c1024_i32 step %c1_i32 iter_args(%arg2 = %cst) -> (tensor<1x1024xf32>)  : i32 {
      %1 = arith.index_cast %arg1 : i32 to index
      %extracted = tensor.extract %arg0[%1] : tensor<1024xf32>
      %inserted = tensor.insert %extracted into %arg2[%c0, %1] : tensor<1x1024xf32>
      scf.yield %inserted : tensor<1x1024xf32>
    }
    %extracted_slice = tensor.extract_slice %0[0, 0] [1, 1024] [1, 1] : tensor<1x1024xf32> to tensor<1024xf32>
    %dps_2 = bufferization.alloc_tensor() : tensor<!plaintext>
    %pt = cheddar.encode %encoder, %extracted_slice, %dps_2 {level = 0 : i64, scale = 0x4270000000000000 : f64} : (!encoder, tensor<1024xf32>, tensor<!plaintext>) -> tensor<!plaintext>
    %dps_3 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct = cheddar.encrypt %ui, %pt, %dps_3 : (!user_interface, tensor<!plaintext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %fe_4 = tensor.empty() : tensor<1x!ciphertext>
    %from_elements = tensor.insert_slice %ct into %fe_4[0] [1] [1] : tensor<!ciphertext> into tensor<1x!ciphertext>
    return %from_elements : tensor<1x!ciphertext>
  }
  func.func @boot__decrypt__result0(%ctx: !context, %encoder: !encoder, %ui: !user_interface, %evk: !eval_key, %arg0: tensor<1x!ciphertext>, %ui_0: !user_interface) -> tensor<1024xf32> attributes {client.dec_func = {func_name = "boot", index = 0 : i64}} {
    %c0 = arith.constant 0 : index
    %c1024_i32 = arith.constant 1024 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32>
    %extracted = tensor.extract_slice %arg0[0] [1] [1] : tensor<1x!ciphertext> to tensor<!ciphertext>
    %dps_5 = bufferization.alloc_tensor() : tensor<!plaintext>
    %pt = cheddar.decrypt %ui, %extracted, %dps_5 : (!user_interface, tensor<!ciphertext>, tensor<!plaintext>) -> tensor<!plaintext>
    %0 = tensor.empty() : tensor<1x1024xf32>
    %1 = cheddar.decode %encoder, %pt, %0 : (!encoder, tensor<!plaintext>, tensor<1x1024xf32>) -> tensor<1x1024xf32>
    %2 = scf.for %arg1 = %c0_i32 to %c1024_i32 step %c1_i32 iter_args(%arg2 = %cst) -> (tensor<1024xf32>)  : i32 {
      %3 = arith.index_cast %arg1 : i32 to index
      %extracted_1 = tensor.extract %1[%c0, %3] : tensor<1x1024xf32>
      %inserted = tensor.insert %extracted_1 into %arg2[%3] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    return %2 : tensor<1024xf32>
  }
}
