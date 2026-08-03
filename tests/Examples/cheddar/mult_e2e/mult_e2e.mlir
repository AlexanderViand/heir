// Element-wise product of two secret 1024-vectors at the cheddar level:
// full-module e2e input (client encrypt/decrypt helpers + compute). Generated
// from a linalg.generic (arith.mulf) source with:
//   heir-opt --annotate-module='backend=cheddar scheme=ckks' \
//            --linalg-canonicalizations \
//            --torch-linalg-to-ckks=ciphertext-degree=1024 --scheme-to-cheddar
// Shallow (one mult): all encodes top-level, so no scale bake is needed.
//
// --scheme-to-cheddar runs cheddar-fuse-ops by default, so this IR uses the
// compound kernels hmult (mult+relinearize[+rescale]) and hrot_add (hrot+add)
// rather than the separate-op sequences.
!ciphertext = !cheddar.ciphertext
!context = !cheddar.context
!encoder = !cheddar.encoder
!eval_key = !cheddar.eval_key
!plaintext = !cheddar.plaintext
!user_interface = !cheddar.user_interface
#layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 1024 = 0 and 0 <= i0 <= 1023 and 0 <= slot <= 1023 }">
#original_type = #tensor_ext.original_type<originalType = tensor<1024xf32>, layout = #layout>
module attributes {backend.cheddar, cheddar.P = array<i64: 1152921504606994433>, cheddar.Q = array<i64: 36028797018652673, 35184372121601>, cheddar.logDefaultScale = 45 : i64, cheddar.logN = 13 : i64, scheme.actual_slot_count = 4096 : i64, scheme.requested_slot_count = 1024 : i64} {
  func.func @mult(%ctx: !context, %encoder: !encoder, %ui: !user_interface, %evk: !eval_key, %arg0: tensor<1x!ciphertext> {tensor_ext.original_type = #original_type}, %arg1: tensor<1x!ciphertext> {tensor_ext.original_type = #original_type}) -> (tensor<1x!ciphertext> {secret.secret, tensor_ext.original_type = #original_type}) {
    %c0 = arith.constant 0 : index
    %extracted = tensor.extract_slice %arg0[0] [1] [1] : tensor<1x!ciphertext> to tensor<!ciphertext>
    %extracted_0 = tensor.extract_slice %arg1[0] [1] [1] : tensor<1x!ciphertext> to tensor<!ciphertext>
    %0 = tensor.empty() : tensor<1x!ciphertext>
    %dps_1 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct = cheddar.hmult %ctx, %extracted, %extracted_0, %evk, %dps_1 : (!context, tensor<!ciphertext>, tensor<!ciphertext>, !eval_key, tensor<!ciphertext>) -> tensor<!ciphertext>
    %inserted = tensor.insert_slice %ct into %0[0] [1] [1] : tensor<!ciphertext> into tensor<1x!ciphertext>
    return %inserted : tensor<1x!ciphertext>
  }
  func.func @mult__encrypt__arg0(%ctx: !context, %encoder: !encoder, %ui: !user_interface, %evk: !eval_key, %arg0: tensor<1024xf32>, %ui_0: !user_interface) -> tensor<1x!ciphertext> attributes {client.enc_func = {func_name = "mult", index = 0 : i64}} {
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
    %pt = cheddar.encode %encoder, %extracted_slice, %dps_2 {level = 1 : i64, scale = 0x42C0000000000000 : f64} : (!encoder, tensor<1024xf32>, tensor<!plaintext>) -> tensor<!plaintext>
    %dps_3 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct = cheddar.encrypt %ui, %pt, %dps_3 : (!user_interface, tensor<!plaintext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %fe_4 = tensor.empty() : tensor<1x!ciphertext>
    %from_elements = tensor.insert_slice %ct into %fe_4[0] [1] [1] : tensor<!ciphertext> into tensor<1x!ciphertext>
    return %from_elements : tensor<1x!ciphertext>
  }
  func.func @mult__encrypt__arg1(%ctx: !context, %encoder: !encoder, %ui: !user_interface, %evk: !eval_key, %arg0: tensor<1024xf32>, %ui_0: !user_interface) -> tensor<1x!ciphertext> attributes {client.enc_func = {func_name = "mult", index = 1 : i64}} {
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
    %dps_5 = bufferization.alloc_tensor() : tensor<!plaintext>
    %pt = cheddar.encode %encoder, %extracted_slice, %dps_5 {level = 1 : i64, scale = 0x42C0000000000000 : f64} : (!encoder, tensor<1024xf32>, tensor<!plaintext>) -> tensor<!plaintext>
    %dps_6 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct = cheddar.encrypt %ui, %pt, %dps_6 : (!user_interface, tensor<!plaintext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %fe_7 = tensor.empty() : tensor<1x!ciphertext>
    %from_elements = tensor.insert_slice %ct into %fe_7[0] [1] [1] : tensor<!ciphertext> into tensor<1x!ciphertext>
    return %from_elements : tensor<1x!ciphertext>
  }
  func.func @mult__decrypt__result0(%ctx: !context, %encoder: !encoder, %ui: !user_interface, %evk: !eval_key, %arg0: tensor<1x!ciphertext>, %ui_0: !user_interface) -> tensor<1024xf32> attributes {client.dec_func = {func_name = "mult", index = 0 : i64}} {
    %c0 = arith.constant 0 : index
    %c1024_i32 = arith.constant 1024 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32>
    %extracted = tensor.extract_slice %arg0[0] [1] [1] : tensor<1x!ciphertext> to tensor<!ciphertext>
    %dps_8 = bufferization.alloc_tensor() : tensor<!plaintext>
    %pt = cheddar.decrypt %ui, %extracted, %dps_8 : (!user_interface, tensor<!ciphertext>, tensor<!plaintext>) -> tensor<!plaintext>
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
