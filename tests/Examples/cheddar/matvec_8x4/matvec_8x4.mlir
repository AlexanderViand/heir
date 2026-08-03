// Matrix-vector product (public 8x4 weight matrix times a secret 4-vector ->
// secret 8-vector) at the cheddar level: full-module e2e input (weight-encoding
// preprocessing + client encrypt/decrypt helpers + compute). Generated from a
// linalg.matvec source with:
//   heir-opt --annotate-module='backend=cheddar scheme=ckks' \
//            --linalg-canonicalizations \
//            --torch-linalg-to-ckks=ciphertext-degree=1024 --scheme-to-cheddar
// Shallow (one mult): all encodes are at the top level, so no scale bake is
// needed (GetScale(top) == nominal 2^45).
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
#layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 8 = 0 and 0 <= i0 <= 7 and 0 <= slot <= 1023 }">
#original_type = #tensor_ext.original_type<originalType = tensor<8xf32>, layout = #layout>
module attributes {backend.cheddar, cheddar.P = array<i64: 1152921504606994433>, cheddar.Q = array<i64: 36028797018652673, 35184372121601>, cheddar.logDefaultScale = 45 : i64, cheddar.logN = 13 : i64, scheme.actual_slot_count = 4096 : i64, scheme.requested_slot_count = 1024 : i64} {
  func.func private @_assign_layout_1727124709149808026(%arg0: tensor<8x4xf32>) -> tensor<4x1024xf32> attributes {client.pack_func = {func_name = "matvec"}} {
    %c1024_i32 = arith.constant 1024 : i32
    %c4_i32 = arith.constant 4 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<4x1024xf32>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = scf.for %arg1 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg2 = %cst) -> (tensor<4x1024xf32>)  : i32 {
      %1 = scf.for %arg3 = %c0_i32 to %c1024_i32 step %c1_i32 iter_args(%arg4 = %arg2) -> (tensor<4x1024xf32>)  : i32 {
        %2 = arith.remsi %arg3, %c8_i32 : i32
        %3 = arith.addi %arg1, %arg3 : i32
        %4 = arith.remsi %3, %c4_i32 : i32
        %5 = arith.index_cast %2 : i32 to index
        %6 = arith.index_cast %4 : i32 to index
        %extracted = tensor.extract %arg0[%5, %6] : tensor<8x4xf32>
        %7 = arith.index_cast %arg1 : i32 to index
        %8 = arith.index_cast %arg3 : i32 to index
        %inserted = tensor.insert %extracted into %arg4[%7, %8] : tensor<4x1024xf32>
        scf.yield %inserted : tensor<4x1024xf32>
      }
      scf.yield %1 : tensor<4x1024xf32>
    }
    return %0 : tensor<4x1024xf32>
  }
  func.func @matvec__preprocessing(%encoder: !encoder, %arg0: tensor<8x4xf32>) -> (tensor<1x!plaintext>, tensor<1x!plaintext>, tensor<1x!plaintext>, tensor<1x!plaintext>) attributes {client.pack_func = {func_name = "matvec"}} {
    %0 = call @_assign_layout_1727124709149808026(%arg0) : (tensor<8x4xf32>) -> tensor<4x1024xf32>
    %extracted_slice = tensor.extract_slice %0[2, 0] [1, 1022] [1, 1] : tensor<4x1024xf32> to tensor<1x1022xf32>
    %extracted_slice_0 = tensor.extract_slice %0[2, 1022] [1, 2] [1, 1] : tensor<4x1024xf32> to tensor<1x2xf32>
    %1 = tensor.empty() : tensor<1x1024xf32>
    %inserted_slice = tensor.insert_slice %extracted_slice into %1[0, 2] [1, 1022] [1, 1] : tensor<1x1022xf32> into tensor<1x1024xf32>
    %inserted_slice_1 = tensor.insert_slice %extracted_slice_0 into %inserted_slice[0, 0] [1, 2] [1, 1] : tensor<1x2xf32> into tensor<1x1024xf32>
    %extracted_slice_2 = tensor.extract_slice %0[3, 0] [1, 1022] [1, 1] : tensor<4x1024xf32> to tensor<1x1022xf32>
    %extracted_slice_3 = tensor.extract_slice %0[3, 1022] [1, 2] [1, 1] : tensor<4x1024xf32> to tensor<1x2xf32>
    %inserted_slice_4 = tensor.insert_slice %extracted_slice_2 into %1[0, 2] [1, 1022] [1, 1] : tensor<1x1022xf32> into tensor<1x1024xf32>
    %inserted_slice_5 = tensor.insert_slice %extracted_slice_3 into %inserted_slice_4[0, 0] [1, 2] [1, 1] : tensor<1x2xf32> into tensor<1x1024xf32>
    %extracted_slice_6 = tensor.extract_slice %0[0, 0] [1, 1024] [1, 1] : tensor<4x1024xf32> to tensor<1024xf32>
    %dps_1 = bufferization.alloc_tensor() : tensor<!plaintext>
    %pt = cheddar.encode %encoder, %extracted_slice_6, %dps_1 {level = 1 : i64, scale = 0x42C0000000000000 : f64} : (!encoder, tensor<1024xf32>, tensor<!plaintext>) -> tensor<!plaintext>
    %extracted_slice_7 = tensor.extract_slice %0[1, 0] [1, 1024] [1, 1] : tensor<4x1024xf32> to tensor<1024xf32>
    %dps_2 = bufferization.alloc_tensor() : tensor<!plaintext>
    %pt_8 = cheddar.encode %encoder, %extracted_slice_7, %dps_2 {level = 1 : i64, scale = 0x42C0000000000000 : f64} : (!encoder, tensor<1024xf32>, tensor<!plaintext>) -> tensor<!plaintext>
    %extracted_slice_9 = tensor.extract_slice %inserted_slice_1[0, 0] [1, 1024] [1, 1] : tensor<1x1024xf32> to tensor<1024xf32>
    %dps_3 = bufferization.alloc_tensor() : tensor<!plaintext>
    %pt_10 = cheddar.encode %encoder, %extracted_slice_9, %dps_3 {level = 1 : i64, scale = 0x42C0000000000000 : f64} : (!encoder, tensor<1024xf32>, tensor<!plaintext>) -> tensor<!plaintext>
    %extracted_slice_11 = tensor.extract_slice %inserted_slice_5[0, 0] [1, 1024] [1, 1] : tensor<1x1024xf32> to tensor<1024xf32>
    %dps_4 = bufferization.alloc_tensor() : tensor<!plaintext>
    %pt_12 = cheddar.encode %encoder, %extracted_slice_11, %dps_4 {level = 1 : i64, scale = 0x42C0000000000000 : f64} : (!encoder, tensor<1024xf32>, tensor<!plaintext>) -> tensor<!plaintext>
    %fe_5 = tensor.empty() : tensor<1x!plaintext>
    %from_elements = tensor.insert_slice %pt into %fe_5[0] [1] [1] : tensor<!plaintext> into tensor<1x!plaintext>
    %fe_6 = tensor.empty() : tensor<1x!plaintext>
    %from_elements_13 = tensor.insert_slice %pt_8 into %fe_6[0] [1] [1] : tensor<!plaintext> into tensor<1x!plaintext>
    %fe_7 = tensor.empty() : tensor<1x!plaintext>
    %from_elements_14 = tensor.insert_slice %pt_10 into %fe_7[0] [1] [1] : tensor<!plaintext> into tensor<1x!plaintext>
    %fe_8 = tensor.empty() : tensor<1x!plaintext>
    %from_elements_15 = tensor.insert_slice %pt_12 into %fe_8[0] [1] [1] : tensor<!plaintext> into tensor<1x!plaintext>
    return %from_elements, %from_elements_13, %from_elements_14, %from_elements_15 : tensor<1x!plaintext>, tensor<1x!plaintext>, tensor<1x!plaintext>, tensor<1x!plaintext>
  }
  func.func @matvec__preprocessed(%ctx: !context, %encoder: !encoder, %ui: !user_interface, %evk: !eval_key, %arg0: tensor<1x!ciphertext>, %arg1: tensor<1x!plaintext>, %arg2: tensor<1x!plaintext>, %arg3: tensor<1x!plaintext>, %arg4: tensor<1x!plaintext>) -> tensor<1x!ciphertext> attributes {client.preprocessed_func = {func_name = "matvec"}} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %extracted = tensor.extract_slice %arg1[0] [1] [1] : tensor<1x!plaintext> to tensor<!plaintext>
    %extracted_0 = tensor.extract_slice %arg2[0] [1] [1] : tensor<1x!plaintext> to tensor<!plaintext>
    %extracted_1 = tensor.extract_slice %arg3[0] [1] [1] : tensor<1x!plaintext> to tensor<!plaintext>
    %extracted_2 = tensor.extract_slice %arg4[0] [1] [1] : tensor<1x!plaintext> to tensor<!plaintext>
    %extracted_3 = tensor.extract_slice %arg0[0] [1] [1] : tensor<1x!ciphertext> to tensor<!ciphertext>
    %dps_9 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct = cheddar.mult_plain %ctx, %extracted_3, %extracted, %dps_9 : (!context, tensor<!ciphertext>, tensor<!plaintext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dps_10 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_4 = cheddar.hrot %ctx, %extracted_3, %dps_10, %c1 : (!context, tensor<!ciphertext>, tensor<!ciphertext>, index) -> tensor<!ciphertext>
    %dps_11 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_5 = cheddar.mult_plain %ctx, %ct_4, %extracted_0, %dps_11 : (!context, tensor<!ciphertext>, tensor<!plaintext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dps_12 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_6 = cheddar.mult_plain %ctx, %extracted_3, %extracted_1, %dps_12 : (!context, tensor<!ciphertext>, tensor<!plaintext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dps_13 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_7 = cheddar.mult_plain %ctx, %ct_4, %extracted_2, %dps_13 : (!context, tensor<!ciphertext>, tensor<!plaintext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dps_14 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_8 = cheddar.add %ctx, %ct_6, %ct_7, %dps_14 : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dps_15 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_9 = cheddar.add %ctx, %ct, %ct_5, %dps_15 : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dps_16 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_10 = cheddar.hrot_add %ctx, %ct_8, %ct_9, %dps_16 {distance = 2 : i64} : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %0 = tensor.empty() : tensor<1x!ciphertext>
    %dps_17 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_11 = cheddar.rescale %ctx, %ct_10, %dps_17 : (!context, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %inserted = tensor.insert_slice %ct_11 into %0[0] [1] [1] : tensor<!ciphertext> into tensor<1x!ciphertext>
    return %inserted : tensor<1x!ciphertext>
  }
  func.func @matvec(%ctx: !context, %encoder: !encoder, %ui: !user_interface, %evk: !eval_key, %arg0: tensor<1x!ciphertext> {tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<4xf32>, layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 4 = 0 and 0 <= i0 <= 3 and 0 <= slot <= 1023 }">>}, %arg1: tensor<8x4xf32>) -> (tensor<1x!ciphertext> {secret.secret, tensor_ext.original_type = #original_type}) {
    %0:4 = call @matvec__preprocessing(%encoder, %arg1) : (!encoder, tensor<8x4xf32>) -> (tensor<1x!plaintext>, tensor<1x!plaintext>, tensor<1x!plaintext>, tensor<1x!plaintext>)
    %1 = call @matvec__preprocessed(%ctx, %encoder, %ui, %evk, %arg0, %0#0, %0#1, %0#2, %0#3) : (!context, !encoder, !user_interface, !eval_key, tensor<1x!ciphertext>, tensor<1x!plaintext>, tensor<1x!plaintext>, tensor<1x!plaintext>, tensor<1x!plaintext>) -> tensor<1x!ciphertext>
    return %1 : tensor<1x!ciphertext>
  }
  func.func @matvec__encrypt__arg0(%ctx: !context, %encoder: !encoder, %ui: !user_interface, %evk: !eval_key, %arg0: tensor<4xf32>, %ui_0: !user_interface) -> tensor<1x!ciphertext> attributes {client.enc_func = {func_name = "matvec", index = 0 : i64}} {
    %c0 = arith.constant 0 : index
    %c4_i32 = arith.constant 4 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %0 = scf.for %arg1 = %c0_i32 to %c1024_i32 step %c1_i32 iter_args(%arg2 = %cst) -> (tensor<1x1024xf32>)  : i32 {
      %1 = arith.remsi %arg1, %c4_i32 : i32
      %2 = arith.index_cast %1 : i32 to index
      %extracted = tensor.extract %arg0[%2] : tensor<4xf32>
      %3 = arith.index_cast %arg1 : i32 to index
      %inserted = tensor.insert %extracted into %arg2[%c0, %3] : tensor<1x1024xf32>
      scf.yield %inserted : tensor<1x1024xf32>
    }
    %extracted_slice = tensor.extract_slice %0[0, 0] [1, 1024] [1, 1] : tensor<1x1024xf32> to tensor<1024xf32>
    %dps_18 = bufferization.alloc_tensor() : tensor<!plaintext>
    %pt = cheddar.encode %encoder, %extracted_slice, %dps_18 {level = 1 : i64, scale = 0x42C0000000000000 : f64} : (!encoder, tensor<1024xf32>, tensor<!plaintext>) -> tensor<!plaintext>
    %dps_19 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct = cheddar.encrypt %ui, %pt, %dps_19 : (!user_interface, tensor<!plaintext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %fe_20 = tensor.empty() : tensor<1x!ciphertext>
    %from_elements = tensor.insert_slice %ct into %fe_20[0] [1] [1] : tensor<!ciphertext> into tensor<1x!ciphertext>
    return %from_elements : tensor<1x!ciphertext>
  }
  func.func @matvec__decrypt__result0(%ctx: !context, %encoder: !encoder, %ui: !user_interface, %evk: !eval_key, %arg0: tensor<1x!ciphertext>, %ui_0: !user_interface) -> tensor<8xf32> attributes {client.dec_func = {func_name = "matvec", index = 0 : i64}} {
    %c0 = arith.constant 0 : index
    %c1024_i32 = arith.constant 1024 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<8xf32>
    %extracted = tensor.extract_slice %arg0[0] [1] [1] : tensor<1x!ciphertext> to tensor<!ciphertext>
    %dps_21 = bufferization.alloc_tensor() : tensor<!plaintext>
    %pt = cheddar.decrypt %ui, %extracted, %dps_21 : (!user_interface, tensor<!ciphertext>, tensor<!plaintext>) -> tensor<!plaintext>
    %0 = tensor.empty() : tensor<1x1024xf32>
    %1 = cheddar.decode %encoder, %pt, %0 : (!encoder, tensor<!plaintext>, tensor<1x1024xf32>) -> tensor<1x1024xf32>
    %2 = scf.for %arg1 = %c0_i32 to %c1024_i32 step %c1_i32 iter_args(%arg2 = %cst) -> (tensor<8xf32>)  : i32 {
      %3 = arith.remsi %arg1, %c8_i32 : i32
      %4 = arith.index_cast %arg1 : i32 to index
      %extracted_1 = tensor.extract %1[%c0, %4] : tensor<1x1024xf32>
      %5 = arith.index_cast %3 : i32 to index
      %inserted = tensor.insert %extracted_1 into %arg2[%5] : tensor<8xf32>
      scf.yield %inserted : tensor<8xf32>
    }
    return %2 : tensor<8xf32>
  }
}
