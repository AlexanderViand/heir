!ciphertext = !cheddar.ciphertext
!context = !cheddar.context
!encoder = !cheddar.encoder
!eval_key = !cheddar.eval_key
!evk_map = !cheddar.evk_map
!plaintext = !cheddar.plaintext
!user_interface = !cheddar.user_interface
module attributes {backend.cheddar, cheddar.P = array<i64: 536641537, 537133057>, cheddar.Q = array<i64: 536903681, 67043329, 67239937, 66813953, 67502081, 66551809, 67731457, 66420737, 68190209, 65929217, 68485121, 68681729, 68976641>, cheddar.logDefaultScale = 26 : i64, cheddar.logN = 14 : i64} {
  func.func @mlpmedium(%ctx: !context, %encoder: !encoder, %ui: !user_interface, %evk: !eval_key, %evk_map: !evk_map, %ct: tensor<!ciphertext>, %arg0: tensor<128x8192xf64> {orion.block_col = 0 : i64, orion.block_row = 0 : i64, orion.layer_name = "fc1", orion.layer_role = "weights", orion.level = 9 : i64}, %arg1: tensor<8192xf64> {orion.layer_name = "fc1", orion.layer_role = "bias", orion.level = 9 : i64}, %arg2: tensor<128x8192xf64> {orion.block_col = 0 : i64, orion.block_row = 0 : i64, orion.layer_name = "fc2", orion.layer_role = "weights", orion.level = 5 : i64}, %arg3: tensor<8192xf64> {orion.layer_name = "fc2", orion.layer_role = "bias", orion.level = 5 : i64}, %arg4: tensor<137x8192xf64> {orion.block_col = 0 : i64, orion.block_row = 0 : i64, orion.layer_name = "fc3", orion.layer_role = "weights", orion.level = 1 : i64}, %arg5: tensor<8192xf64> {orion.layer_name = "fc3", orion.layer_role = "bias", orion.level = 1 : i64}) -> tensor<!ciphertext> {
    %dps_1 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_0 = cheddar.linear_transform %ctx, %ct, %evk_map, %arg0, %dps_1 {bs = 16 : i64, diagonal_indices = array<i32: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127>, gs = 8 : i64, level = 9 : i64} : (!context, tensor<!ciphertext>, !evk_map, tensor<128x8192xf64>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dps_2 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_2 = cheddar.hrot_add %ctx, %ct_0, %ct_0, %dps_2 {distance = 4096 : i32} : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dps_3 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_3 = cheddar.hrot_add %ctx, %ct_2, %ct_2, %dps_3 {distance = 2048 : i32} : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dps_4 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_4 = cheddar.hrot_add %ctx, %ct_3, %ct_3, %dps_4 {distance = 1024 : i32} : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dps_5 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_5 = cheddar.hrot_add %ctx, %ct_4, %ct_4, %dps_5 {distance = 512 : i32} : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dps_6 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_6 = cheddar.hrot_add %ctx, %ct_5, %ct_5, %dps_6 {distance = 256 : i32} : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dps_7 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_7 = cheddar.hrot_add %ctx, %ct_6, %ct_6, %dps_7 {distance = 128 : i32} : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dps_8 = bufferization.alloc_tensor() : tensor<!plaintext>
    %pt = cheddar.encode %encoder, %arg1, %dps_8 {level = 8 : i64, scale = 0x4190195E36D35D53 : f64} : (!encoder, tensor<8192xf64>, tensor<!plaintext>) -> tensor<!plaintext>
    %dps_9 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_8 = cheddar.add_plain %ctx, %ct_7, %pt, %dps_9 : (!context, tensor<!ciphertext>, tensor<!plaintext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dps_10 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_9 = cheddar.eval_poly %ctx, %ct_8, %evk_map, %dps_10 {coefficients = [1.5145835929336107, 2.5613617958306212, 1.1268815055871191, -0.044593940940794094, -0.20397100102388641, 0.021439818106555666, 0.047636893102044199, -0.0095249447540559865], level = 8 : i64, outputLevel = 5 : i64} : (!context, tensor<!ciphertext>, !evk_map, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dps_11 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_10 = cheddar.linear_transform %ctx, %ct_9, %evk_map, %arg2, %dps_11 {bs = 16 : i64, diagonal_indices = array<i32: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127>, gs = 8 : i64, level = 5 : i64} : (!context, tensor<!ciphertext>, !evk_map, tensor<128x8192xf64>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dps_12 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_12 = cheddar.hrot_add %ctx, %ct_10, %ct_10, %dps_12 {distance = 4096 : i32} : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dps_13 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_13 = cheddar.hrot_add %ctx, %ct_12, %ct_12, %dps_13 {distance = 2048 : i32} : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dps_14 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_14 = cheddar.hrot_add %ctx, %ct_13, %ct_13, %dps_14 {distance = 1024 : i32} : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dps_15 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_15 = cheddar.hrot_add %ctx, %ct_14, %ct_14, %dps_15 {distance = 512 : i32} : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dps_16 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_16 = cheddar.hrot_add %ctx, %ct_15, %ct_15, %dps_16 {distance = 256 : i32} : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dps_17 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_17 = cheddar.hrot_add %ctx, %ct_16, %ct_16, %dps_17 {distance = 128 : i32} : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dps_18 = bufferization.alloc_tensor() : tensor<!plaintext>
    %pt_18 = cheddar.encode %encoder, %arg3, %dps_18 {level = 4 : i64, scale = 0x4190083655AF019A : f64} : (!encoder, tensor<8192xf64>, tensor<!plaintext>) -> tensor<!plaintext>
    %dps_19 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_19 = cheddar.add_plain %ctx, %ct_17, %pt_18, %dps_19 : (!context, tensor<!ciphertext>, tensor<!plaintext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dps_20 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_20 = cheddar.eval_poly %ctx, %ct_19, %evk_map, %dps_20 {coefficients = [1.182939746949178, 2.0927927088497427, 0.93284570275258571, -0.024128477414385958, -0.14747130773352449, 0.0091007691604530066, 0.029518955234313079, -0.0032715783141000839], level = 4 : i64, outputLevel = 1 : i64} : (!context, tensor<!ciphertext>, !evk_map, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dps_21 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_21 = cheddar.linear_transform %ctx, %ct_20, %evk_map, %arg4, %dps_21 {bs = 8192 : i64, diagonal_indices = array<i32: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 8183, 8184, 8185, 8186, 8187, 8188, 8189, 8190, 8191>, gs = 1 : i64, level = 1 : i64} : (!context, tensor<!ciphertext>, !evk_map, tensor<137x8192xf64>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dps_22 = bufferization.alloc_tensor() : tensor<!plaintext>
    %pt_23 = cheddar.encode %encoder, %arg5, %dps_22 {level = 0 : i64, scale = 0x4190000000000000 : f64} : (!encoder, tensor<8192xf64>, tensor<!plaintext>) -> tensor<!plaintext>
    %dps_23 = bufferization.alloc_tensor() : tensor<!ciphertext>
    %ct_24 = cheddar.add_plain %ctx, %ct_21, %pt_23, %dps_23 : (!context, tensor<!ciphertext>, tensor<!plaintext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    return %ct_24 : tensor<!ciphertext>
  }
}
