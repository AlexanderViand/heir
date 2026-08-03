// The Orion MLP, lowered from the Orion-extension IR that the orion-to-lattigo
// path emits: three orion.linear_transform ops (the FC layers, with diagonal-
// packed weights as cleartext operands + BSGS) and Quad activations, versus the
// unrolled rotate-and-sum matvecs that HEIR's own --torch-linalg-to-ckks
// produces for the orion_mlp test. Exercises the Orion -> CHEDDAR patterns
// (orion.linear_transform -> cheddar.linear_transform, the EvkMap threaded as a
// contextual arg).
//
// Generation (orion CKKS-level output -> cheddar), then hand-fixed at the
// cheddar-IR level like the other examples:
//   1. The orion frontend emits inverse_canonical_encoding scaling_factor as the
//      LOG value (26 / 52); HEIR expects the actual 2^k, so rewrite 26->2^26,
//      52->2^52.
//   2. heir-opt --annotate-module='backend=cheddar scheme=ckks' --scheme-to-cheddar
//   3. bias-encode levels set to the post-LT-rescale level (4 / 2 / 0) and the
//      scales baked to CHEDDAR's exact GetScale(level) for this 6-prime, 2^26
//      param set (scale_dump.cpp); the LT diagonal plaintexts are encoded by
//      CHEDDAR internally at GetScale(level), so they need no bake.
// Params: logN=13, Q=6 primes, P=2 primes, logDefaultScale=26, max_level 5.
!ciphertext = !cheddar.ciphertext
!context = !cheddar.context
!encoder = !cheddar.encoder
!eval_key = !cheddar.eval_key
!evk_map = !cheddar.evk_map
!plaintext = !cheddar.plaintext
!user_interface = !cheddar.user_interface
module attributes {backend.cheddar, cheddar.P = array<i64: 536952833, 536690689>, cheddar.Q = array<i64: 536903681, 67043329, 66994177, 67239937, 66961409, 66813953>, cheddar.logDefaultScale = 26 : i64, cheddar.logN = 13 : i64} {
  func.func @orionmlp(%ctx: !context, %encoder: !encoder, %ui: !user_interface, %evk: !eval_key, %evk_map: !evk_map, %ct: tensor<!ciphertext>, %arg0: tensor<128x4096xf64> {orion.block_col = 0 : i64, orion.block_row = 0 : i64, orion.layer_name = "fc1", orion.layer_role = "weights", orion.level = 5 : i64}, %arg1: tensor<4096xf64> {orion.layer_name = "fc1", orion.layer_role = "bias", orion.level = 5 : i64}, %arg2: tensor<128x4096xf64> {orion.block_col = 0 : i64, orion.block_row = 0 : i64, orion.layer_name = "fc2", orion.layer_role = "weights", orion.level = 3 : i64}, %arg3: tensor<4096xf64> {orion.layer_name = "fc2", orion.layer_role = "bias", orion.level = 3 : i64}, %arg4: tensor<137x4096xf64> {orion.block_col = 0 : i64, orion.block_row = 0 : i64, orion.layer_name = "fc3", orion.layer_role = "weights", orion.level = 1 : i64}, %arg5: tensor<4096xf64> {orion.layer_name = "fc3", orion.layer_role = "bias", orion.level = 1 : i64}) -> tensor<!ciphertext> {
    %dps_1 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_0 = cheddar.linear_transform %ctx, %ct, %evk_map, %arg0, %dps_1 {diagonal_indices = array<i32: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127>, level = 5 : i64, bs = 16 : i64, gs = 8 : i64} : (!context, tensor<!ciphertext>, !evk_map, tensor<128x4096xf64>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dps_2 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_2 = cheddar.hrot_add %ctx, %ct_0, %ct_0, %dps_2 {distance = 2048 : i32} : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dps_3 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_3 = cheddar.hrot_add %ctx, %ct_2, %ct_2, %dps_3 {distance = 1024 : i32} : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dps_4 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_4 = cheddar.hrot_add %ctx, %ct_3, %ct_3, %dps_4 {distance = 512 : i32} : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dps_5 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_5 = cheddar.hrot_add %ctx, %ct_4, %ct_4, %dps_5 {distance = 256 : i32} : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dps_6 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_6 = cheddar.hrot_add %ctx, %ct_5, %ct_5, %dps_6 {distance = 128 : i32} : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dps_7 = bufferization.alloc_tensor() : tensor<!plaintext>
    %pt = cheddar.encode %encoder, %arg1, %dps_7 {level = 4 : i64, scale = 0x418FF8BCDFD9B6C7 : f64} : (!encoder, tensor<4096xf64>, tensor<!plaintext>) -> tensor<!plaintext>
    %dps_8 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_7 = cheddar.add_plain %ctx, %ct_6, %pt, %dps_8 : (!context, tensor<!ciphertext>, tensor<!plaintext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dps_9 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_8 = cheddar.hmult %ctx, %ct_7, %ct_7, %evk, %dps_9 : (!context, tensor<!ciphertext>, tensor<!ciphertext>, !eval_key, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dps_10 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_9 = cheddar.linear_transform %ctx, %ct_8, %evk_map, %arg2, %dps_10 {diagonal_indices = array<i32: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127>, level = 3 : i64, bs = 16 : i64, gs = 8 : i64} : (!context, tensor<!ciphertext>, !evk_map, tensor<128x4096xf64>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dps_11 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_11 = cheddar.hrot_add %ctx, %ct_9, %ct_9, %dps_11 {distance = 2048 : i32} : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dps_12 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_12 = cheddar.hrot_add %ctx, %ct_11, %ct_11, %dps_12 {distance = 1024 : i32} : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dps_13 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_13 = cheddar.hrot_add %ctx, %ct_12, %ct_12, %dps_13 {distance = 512 : i32} : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dps_14 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_14 = cheddar.hrot_add %ctx, %ct_13, %ct_13, %dps_14 {distance = 256 : i32} : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dps_15 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_15 = cheddar.hrot_add %ctx, %ct_14, %ct_14, %dps_15 {distance = 128 : i32} : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dps_16 = bufferization.alloc_tensor() : tensor<!plaintext>
    %pt_16 = cheddar.encode %encoder, %arg3, %dps_16 {level = 2 : i64, scale = 0x418FF6FF81E56895 : f64} : (!encoder, tensor<4096xf64>, tensor<!plaintext>) -> tensor<!plaintext>
    %dps_17 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_17 = cheddar.add_plain %ctx, %ct_15, %pt_16, %dps_17 : (!context, tensor<!ciphertext>, tensor<!plaintext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dps_18 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_18 = cheddar.hmult %ctx, %ct_17, %ct_17, %evk, %dps_18 : (!context, tensor<!ciphertext>, tensor<!ciphertext>, !eval_key, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dps_19 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_19 = cheddar.linear_transform %ctx, %ct_18, %evk_map, %arg4, %dps_19 {diagonal_indices = array<i32: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 4087, 4088, 4089, 4090, 4091, 4092, 4093, 4094, 4095>, level = 1 : i64, bs = 4096 : i64, gs = 1 : i64} : (!context, tensor<!ciphertext>, !evk_map, tensor<137x4096xf64>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dps_20 = bufferization.alloc_tensor() : tensor<!plaintext>
    %pt_21 = cheddar.encode %encoder, %arg5, %dps_20 {level = 0 : i64, scale = 0x4190000000000000 : f64} : (!encoder, tensor<4096xf64>, tensor<!plaintext>) -> tensor<!plaintext>
    %dps_21 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_22 = cheddar.add_plain %ctx, %ct_19, %pt_21, %dps_21 : (!context, tensor<!ciphertext>, tensor<!plaintext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    return %ct_22 : tensor<!ciphertext>
  }
}
