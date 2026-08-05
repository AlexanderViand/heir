// Arithmetic grab-bag on two secret 1024-vectors at the cheddar level -- ct+ct,
// then ct-ct, then ct*plaintext(2.0), then ct+plaintext(0.5), so the result is
// 2*a + 0.5. Full-module e2e input (client encrypt/decrypt helpers + a
// preprocessing that encodes the 2.0 and 0.5 constants + compute).
//
// Destination-passing form: each payload-producing cheddar op takes an explicit
// `bufferization.alloc_tensor` destination (its `outs` init); a scalar payload
// is a rank-0 `tensor<!cheddar.X>`, extracted from / inserted into the
// `tensor<1x!cheddar.X>` packing container via rank-reducing slice ops.
//
// The plaintext multiply makes this depth-1 (max_level 1, two Q primes) -- a
// single Q prime is not a viable CHEDDAR modulus chain. The 2.0 is encoded at
// the top level (level 1); the product is rescaled back to the canonical scale
// before adding the 0.5, which is encoded at level 0 -- so every op sees
// matching per-level GetScale scales, no scale bake or tolerance needed.
!ciphertext = !cheddar.ciphertext
!context = !cheddar.context
!encoder = !cheddar.encoder
!eval_key = !cheddar.eval_key
!plaintext = !cheddar.plaintext
!user_interface = !cheddar.user_interface
#layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 1024 = 0 and 0 <= i0 <= 1023 and 0 <= slot <= 1023 }">
#original_type = #tensor_ext.original_type<originalType = tensor<1024xf32>, layout = #layout>
module attributes {backend.cheddar, cheddar.P = array<i64: 1152921504606994433>, cheddar.Q = array<i64: 36028797018652673, 35184372121601>, cheddar.logDefaultScale = 45 : i64, cheddar.logN = 13 : i64, scheme.actual_slot_count = 4096 : i64, scheme.requested_slot_count = 1024 : i64} {
  func.func @add__preprocessing(%encoder: !encoder) -> (tensor<1x!plaintext>, tensor<1x!plaintext>) attributes {client.pack_func = {func_name = "add"}} {
    %cst = arith.constant dense<2.000000e+00> : tensor<1024xf32>
    %cst_0 = arith.constant dense<5.000000e-01> : tensor<1024xf32>
    %d_pt = bufferization.alloc_tensor() : tensor<!plaintext>
    %pt = cheddar.encode %encoder, %cst, %d_pt {level = 1 : i64, scale = 0x42C0000000000000 : f64} : (!encoder, tensor<1024xf32>, tensor<!plaintext>) -> tensor<!plaintext>
    %d_pt_0 = bufferization.alloc_tensor() : tensor<!plaintext>
    %pt_1 = cheddar.encode %encoder, %cst_0, %d_pt_0 {level = 0 : i64, scale = 0x42C0000000000000 : f64} : (!encoder, tensor<1024xf32>, tensor<!plaintext>) -> tensor<!plaintext>
    %e0 = tensor.empty() : tensor<1x!plaintext>
    %from_elements = tensor.insert_slice %pt_1 into %e0[0] [1] [1] : tensor<!plaintext> into tensor<1x!plaintext>
    %e1 = tensor.empty() : tensor<1x!plaintext>
    %from_elements_2 = tensor.insert_slice %pt into %e1[0] [1] [1] : tensor<!plaintext> into tensor<1x!plaintext>
    return %from_elements, %from_elements_2 : tensor<1x!plaintext>, tensor<1x!plaintext>
  }
  func.func @add__preprocessed(%ctx: !context, %encoder: !encoder, %ui: !user_interface, %evk: !eval_key, %arg0: tensor<1x!ciphertext>, %arg1: tensor<1x!ciphertext>, %arg2: tensor<1x!plaintext>, %arg3: tensor<1x!plaintext>) -> tensor<1x!ciphertext> attributes {client.preprocessed_func = {func_name = "add"}} {
    %pt_05 = tensor.extract_slice %arg2[0] [1] [1] : tensor<1x!plaintext> to tensor<!plaintext>
    %pt_20 = tensor.extract_slice %arg3[0] [1] [1] : tensor<1x!plaintext> to tensor<!plaintext>
    %a = tensor.extract_slice %arg0[0] [1] [1] : tensor<1x!ciphertext> to tensor<!ciphertext>
    %b = tensor.extract_slice %arg1[0] [1] [1] : tensor<1x!ciphertext> to tensor<!ciphertext>
    %d0 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct = cheddar.add %ctx, %a, %b, %d0 : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %d1 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_3 = cheddar.sub %ctx, %ct, %b, %d1 : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %d2 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_4 = cheddar.mult_plain %ctx, %ct_3, %pt_20, %d2 : (!context, tensor<!ciphertext>, tensor<!plaintext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %d3 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_5 = cheddar.rescale %ctx, %ct_4, %d3 : (!context, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %d4 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_6 = cheddar.add_plain %ctx, %ct_5, %pt_05, %d4 : (!context, tensor<!ciphertext>, tensor<!plaintext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %e = tensor.empty() : tensor<1x!ciphertext>
    %inserted = tensor.insert_slice %ct_6 into %e[0] [1] [1] : tensor<!ciphertext> into tensor<1x!ciphertext>
    return %inserted : tensor<1x!ciphertext>
  }
  func.func @add(%ctx: !context, %encoder: !encoder, %ui: !user_interface, %evk: !eval_key, %arg0: tensor<1x!ciphertext> {tensor_ext.original_type = #original_type}, %arg1: tensor<1x!ciphertext> {tensor_ext.original_type = #original_type}) -> (tensor<1x!ciphertext> {tensor_ext.original_type = #original_type}) {
    %0:2 = call @add__preprocessing(%encoder) : (!encoder) -> (tensor<1x!plaintext>, tensor<1x!plaintext>)
    %1 = call @add__preprocessed(%ctx, %encoder, %ui, %evk, %arg0, %arg1, %0#0, %0#1) : (!context, !encoder, !user_interface, !eval_key, tensor<1x!ciphertext>, tensor<1x!ciphertext>, tensor<1x!plaintext>, tensor<1x!plaintext>) -> tensor<1x!ciphertext>
    return %1 : tensor<1x!ciphertext>
  }
  func.func @add__encrypt__arg0(%ctx: !context, %encoder: !encoder, %ui: !user_interface, %evk: !eval_key, %arg0: tensor<1024xf32>, %ui_0: !user_interface) -> tensor<1x!ciphertext> attributes {client.enc_func = {func_name = "add", index = 0 : i64}} {
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
    %d_pt = bufferization.alloc_tensor() : tensor<!plaintext>
    %pt = cheddar.encode %encoder, %extracted_slice, %d_pt {level = 1 : i64, scale = 0x42C0000000000000 : f64} : (!encoder, tensor<1024xf32>, tensor<!plaintext>) -> tensor<!plaintext>
    %d_ct = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct = cheddar.encrypt %ui, %pt, %d_ct : (!user_interface, tensor<!plaintext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %e = tensor.empty() : tensor<1x!ciphertext>
    %from_elements = tensor.insert_slice %ct into %e[0] [1] [1] : tensor<!ciphertext> into tensor<1x!ciphertext>
    return %from_elements : tensor<1x!ciphertext>
  }
  func.func @add__encrypt__arg1(%ctx: !context, %encoder: !encoder, %ui: !user_interface, %evk: !eval_key, %arg0: tensor<1024xf32>, %ui_0: !user_interface) -> tensor<1x!ciphertext> attributes {client.enc_func = {func_name = "add", index = 1 : i64}} {
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
    %d_pt = bufferization.alloc_tensor() : tensor<!plaintext>
    %pt = cheddar.encode %encoder, %extracted_slice, %d_pt {level = 1 : i64, scale = 0x42C0000000000000 : f64} : (!encoder, tensor<1024xf32>, tensor<!plaintext>) -> tensor<!plaintext>
    %d_ct = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct = cheddar.encrypt %ui, %pt, %d_ct : (!user_interface, tensor<!plaintext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %e = tensor.empty() : tensor<1x!ciphertext>
    %from_elements = tensor.insert_slice %ct into %e[0] [1] [1] : tensor<!ciphertext> into tensor<1x!ciphertext>
    return %from_elements : tensor<1x!ciphertext>
  }
  func.func @add__decrypt__result0(%ctx: !context, %encoder: !encoder, %ui: !user_interface, %evk: !eval_key, %arg0: tensor<1x!ciphertext>, %ui_0: !user_interface) -> tensor<1024xf32> attributes {client.dec_func = {func_name = "add", index = 0 : i64}} {
    %c0 = arith.constant 0 : index
    %c1024_i32 = arith.constant 1024 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32>
    %a_ct = tensor.extract_slice %arg0[0] [1] [1] : tensor<1x!ciphertext> to tensor<!ciphertext>
    %d_pt = bufferization.alloc_tensor() : tensor<!plaintext>
    %pt = cheddar.decrypt %ui, %a_ct, %d_pt : (!user_interface, tensor<!ciphertext>, tensor<!plaintext>) -> tensor<!plaintext>
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
