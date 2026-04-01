!ct = !cheddar.ciphertext
!ctx = !cheddar.context
!encoder = !cheddar.encoder
!evk = !cheddar.eval_key
!evk_map = !cheddar.evk_map
!pt = !cheddar.plaintext
!ui = !cheddar.user_interface
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [536903681, 67043329, 66994177, 67239937, 66961409, 66813953], P = [536952833, 536690689], logDefaultScale = 26>, scheme.ckks} {
  func.func @mlp(%ctx: !ctx, %encoder: !encoder, %ui: !ui, %ct: !ct, %arg0: tensor<128x4096xf64> {orion.block_col = 0 : i64, orion.block_row = 0 : i64, orion.layer_name = "fc1", orion.layer_role = "weights", orion.level = 5 : i64}, %arg1: tensor<4096xf64> {orion.layer_name = "fc1", orion.layer_role = "bias", orion.level = 5 : i64}, %arg2: tensor<128x4096xf64> {orion.block_col = 0 : i64, orion.block_row = 0 : i64, orion.layer_name = "fc2", orion.layer_role = "weights", orion.level = 3 : i64}, %arg3: tensor<4096xf64> {orion.layer_name = "fc2", orion.layer_role = "bias", orion.level = 3 : i64}, %arg4: tensor<137x4096xf64> {orion.block_col = 0 : i64, orion.block_row = 0 : i64, orion.layer_name = "fc3", orion.layer_role = "weights", orion.level = 1 : i64}, %arg5: tensor<4096xf64> {orion.layer_name = "fc3", orion.layer_role = "bias", orion.level = 1 : i64}) -> !ct {
    %evk_map = cheddar.get_evk_map %ui : (!ui) -> !evk_map
    %ct_0 = cheddar.linear_transform %ctx, %ct, %evk_map, %arg0 {diagonal_indices = array<i32: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127>, level = 5 : i32, logBabyStepGiantStepRatio = 2 : i64} : (!ctx, !ct, !evk_map, tensor<128x4096xf64>) -> !ct
    %evk = cheddar.get_rot_key %ui {distance = 2048 : i32} : (!ui) -> !evk
    %ct_1 = cheddar.hrot_add %ctx, %ct_0, %ct_0, %evk {distance = 2048 : i32} : (!ctx, !ct, !ct, !evk) -> !ct
    %evk_2 = cheddar.get_rot_key %ui {distance = 1024 : i32} : (!ui) -> !evk
    %ct_3 = cheddar.hrot_add %ctx, %ct_1, %ct_1, %evk_2 {distance = 1024 : i32} : (!ctx, !ct, !ct, !evk) -> !ct
    %evk_4 = cheddar.get_rot_key %ui {distance = 512 : i32} : (!ui) -> !evk
    %ct_5 = cheddar.hrot_add %ctx, %ct_3, %ct_3, %evk_4 {distance = 512 : i32} : (!ctx, !ct, !ct, !evk) -> !ct
    %evk_6 = cheddar.get_rot_key %ui {distance = 256 : i32} : (!ui) -> !evk
    %ct_7 = cheddar.hrot_add %ctx, %ct_5, %ct_5, %evk_6 {distance = 256 : i32} : (!ctx, !ct, !ct, !evk) -> !ct
    %evk_8 = cheddar.get_rot_key %ui {distance = 128 : i32} : (!ui) -> !evk
    %ct_9 = cheddar.hrot_add %ctx, %ct_7, %ct_7, %evk_8 {distance = 128 : i32} : (!ctx, !ct, !ct, !evk) -> !ct
    %pt = cheddar.encode %encoder, %arg1 {level = 0 : i64, scale = 1099511627776 : i64} : (!encoder, tensor<4096xf64>) -> !pt
    %ct_10 = cheddar.add_plain %ctx, %ct_9, %pt : (!ctx, !ct, !pt) -> !ct
    %evk_11 = cheddar.get_mult_key %ui : (!ui) -> !evk
    %ct_12 = cheddar.hmult %ctx, %ct_10, %ct_10, %evk_11 : (!ctx, !ct, !ct, !evk) -> !ct
    %evk_map_13 = cheddar.get_evk_map %ui : (!ui) -> !evk_map
    %ct_14 = cheddar.linear_transform %ctx, %ct_12, %evk_map_13, %arg2 {diagonal_indices = array<i32: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127>, level = 3 : i32, logBabyStepGiantStepRatio = 2 : i64} : (!ctx, !ct, !evk_map, tensor<128x4096xf64>) -> !ct
    %evk_15 = cheddar.get_rot_key %ui {distance = 2048 : i32} : (!ui) -> !evk
    %ct_16 = cheddar.hrot_add %ctx, %ct_14, %ct_14, %evk_15 {distance = 2048 : i32} : (!ctx, !ct, !ct, !evk) -> !ct
    %evk_17 = cheddar.get_rot_key %ui {distance = 1024 : i32} : (!ui) -> !evk
    %ct_18 = cheddar.hrot_add %ctx, %ct_16, %ct_16, %evk_17 {distance = 1024 : i32} : (!ctx, !ct, !ct, !evk) -> !ct
    %evk_19 = cheddar.get_rot_key %ui {distance = 512 : i32} : (!ui) -> !evk
    %ct_20 = cheddar.hrot_add %ctx, %ct_18, %ct_18, %evk_19 {distance = 512 : i32} : (!ctx, !ct, !ct, !evk) -> !ct
    %evk_21 = cheddar.get_rot_key %ui {distance = 256 : i32} : (!ui) -> !evk
    %ct_22 = cheddar.hrot_add %ctx, %ct_20, %ct_20, %evk_21 {distance = 256 : i32} : (!ctx, !ct, !ct, !evk) -> !ct
    %evk_23 = cheddar.get_rot_key %ui {distance = 128 : i32} : (!ui) -> !evk
    %ct_24 = cheddar.hrot_add %ctx, %ct_22, %ct_22, %evk_23 {distance = 128 : i32} : (!ctx, !ct, !ct, !evk) -> !ct
    %pt_25 = cheddar.encode %encoder, %arg3 {level = 0 : i64, scale = 1099511627776 : i64} : (!encoder, tensor<4096xf64>) -> !pt
    %ct_26 = cheddar.add_plain %ctx, %ct_24, %pt_25 : (!ctx, !ct, !pt) -> !ct
    %evk_27 = cheddar.get_mult_key %ui : (!ui) -> !evk
    %ct_28 = cheddar.hmult %ctx, %ct_26, %ct_26, %evk_27 : (!ctx, !ct, !ct, !evk) -> !ct
    %evk_map_29 = cheddar.get_evk_map %ui : (!ui) -> !evk_map
    %ct_30 = cheddar.linear_transform %ctx, %ct_28, %evk_map_29, %arg4 {diagonal_indices = array<i32: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 4087, 4088, 4089, 4090, 4091, 4092, 4093, 4094, 4095>, level = 1 : i32, logBabyStepGiantStepRatio = 2 : i64} : (!ctx, !ct, !evk_map, tensor<137x4096xf64>) -> !ct
    %pt_31 = cheddar.encode %encoder, %arg5 {level = 0 : i64, scale = 1099511627776 : i64} : (!encoder, tensor<4096xf64>) -> !pt
    %ct_32 = cheddar.add_plain %ctx, %ct_30, %pt_31 : (!ctx, !ct, !pt) -> !ct
    return %ct_32 : !ct
  }
}
