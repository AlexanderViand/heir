!ct = !cheddar.ciphertext
!ctx = !cheddar.context
!encoder = !cheddar.encoder
!evk = !cheddar.eval_key
!evk_map = !cheddar.evk_map
!pt = !cheddar.plaintext
!ui = !cheddar.user_interface
module attributes {cheddar.P = array<i64: 1152921504607338497>, cheddar.Q = array<i64: 1073643521, 67731457, 66813953, 67502081, 67043329, 67239937>, cheddar.logDefaultScale = 26 : i64, cheddar.logN = 14 : i64, ckks.reduced_error = false, ckks.scale_policy = "nominal", heir.level_offset = 3 : i64, scheme.actual_slot_count = 8192 : i64, scheme.requested_slot_count = 4096 : i64} {
  func.func @mlp(%ctx: !ctx, %encoder: !encoder, %ui: !ui, %ct: !ct, %arg0: tensor<128x4096xf64>, %arg1: tensor<4096xf64>, %arg2: tensor<128x4096xf64>, %arg3: tensor<4096xf64>, %arg4: tensor<137x4096xf64>, %arg5: tensor<4096xf64>) -> !ct {
    %c2048 = arith.constant 2048 : index
    %c1024 = arith.constant 1024 : index
    %c512 = arith.constant 512 : index
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %evk_map = cheddar.get_evk_map %ui : (!ui) -> !evk_map
    %ct_0 = cheddar.linear_transform %ctx, %ct, %evk_map, %arg0 {diagonal_indices = array<i32: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127>, level = 5 : i64, logBabyStepGiantStepRatio = 1 : i64} : (!ctx, !ct, !evk_map, tensor<128x4096xf64>) -> !ct
    %evk = cheddar.get_rot_key %ui {distance = -1 : i64} : (!ui) -> !evk
    %ct_1 = cheddar.hrot %ctx, %ct_0, %evk, %c2048 : (!ctx, !ct, !evk, index) -> !ct
    %ct_2 = cheddar.add %ctx, %ct_1, %ct_0 : (!ctx, !ct, !ct) -> !ct
    %evk_3 = cheddar.get_rot_key %ui {distance = -1 : i64} : (!ui) -> !evk
    %ct_4 = cheddar.hrot %ctx, %ct_2, %evk_3, %c1024 : (!ctx, !ct, !evk, index) -> !ct
    %ct_5 = cheddar.add %ctx, %ct_4, %ct_2 : (!ctx, !ct, !ct) -> !ct
    %evk_6 = cheddar.get_rot_key %ui {distance = -1 : i64} : (!ui) -> !evk
    %ct_7 = cheddar.hrot %ctx, %ct_5, %evk_6, %c512 : (!ctx, !ct, !evk, index) -> !ct
    %ct_8 = cheddar.add %ctx, %ct_7, %ct_5 : (!ctx, !ct, !ct) -> !ct
    %evk_9 = cheddar.get_rot_key %ui {distance = -1 : i64} : (!ui) -> !evk
    %ct_10 = cheddar.hrot %ctx, %ct_8, %evk_9, %c256 : (!ctx, !ct, !evk, index) -> !ct
    %ct_11 = cheddar.add %ctx, %ct_10, %ct_8 : (!ctx, !ct, !ct) -> !ct
    %evk_12 = cheddar.get_rot_key %ui {distance = -1 : i64} : (!ui) -> !evk
    %ct_13 = cheddar.hrot %ctx, %ct_11, %evk_12, %c128 : (!ctx, !ct, !evk, index) -> !ct
    %ct_14 = cheddar.add %ctx, %ct_13, %ct_11 : (!ctx, !ct, !ct) -> !ct
    %pt = cheddar.encode %encoder, %arg1 {level = 4 : i64, scale = 26 : i64} : (!encoder, tensor<4096xf64>) -> !pt
    %ct_15 = cheddar.add_plain %ctx, %ct_14, %pt : (!ctx, !ct, !pt) -> !ct
    %ct_16 = cheddar.mult %ctx, %ct_15, %ct_15 : (!ctx, !ct, !ct) -> !ct
    %evk_17 = cheddar.get_mult_key %ui : (!ui) -> !evk
    %ct_18 = cheddar.relinearize %ctx, %ct_16, %evk_17 : (!ctx, !ct, !evk) -> !ct
    %evk_map_19 = cheddar.get_evk_map %ui : (!ui) -> !evk_map
    %ct_20 = cheddar.linear_transform %ctx, %ct_18, %evk_map_19, %arg2 {diagonal_indices = array<i32: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127>, level = 4 : i64, logBabyStepGiantStepRatio = 1 : i64} : (!ctx, !ct, !evk_map, tensor<128x4096xf64>) -> !ct
    %evk_21 = cheddar.get_rot_key %ui {distance = -1 : i64} : (!ui) -> !evk
    %ct_22 = cheddar.hrot %ctx, %ct_20, %evk_21, %c2048 : (!ctx, !ct, !evk, index) -> !ct
    %ct_23 = cheddar.add %ctx, %ct_22, %ct_20 : (!ctx, !ct, !ct) -> !ct
    %evk_24 = cheddar.get_rot_key %ui {distance = -1 : i64} : (!ui) -> !evk
    %ct_25 = cheddar.hrot %ctx, %ct_23, %evk_24, %c1024 : (!ctx, !ct, !evk, index) -> !ct
    %ct_26 = cheddar.add %ctx, %ct_25, %ct_23 : (!ctx, !ct, !ct) -> !ct
    %evk_27 = cheddar.get_rot_key %ui {distance = -1 : i64} : (!ui) -> !evk
    %ct_28 = cheddar.hrot %ctx, %ct_26, %evk_27, %c512 : (!ctx, !ct, !evk, index) -> !ct
    %ct_29 = cheddar.add %ctx, %ct_28, %ct_26 : (!ctx, !ct, !ct) -> !ct
    %evk_30 = cheddar.get_rot_key %ui {distance = -1 : i64} : (!ui) -> !evk
    %ct_31 = cheddar.hrot %ctx, %ct_29, %evk_30, %c256 : (!ctx, !ct, !evk, index) -> !ct
    %ct_32 = cheddar.add %ctx, %ct_31, %ct_29 : (!ctx, !ct, !ct) -> !ct
    %evk_33 = cheddar.get_rot_key %ui {distance = -1 : i64} : (!ui) -> !evk
    %ct_34 = cheddar.hrot %ctx, %ct_32, %evk_33, %c128 : (!ctx, !ct, !evk, index) -> !ct
    %ct_35 = cheddar.add %ctx, %ct_34, %ct_32 : (!ctx, !ct, !ct) -> !ct
    %pt_36 = cheddar.encode %encoder, %arg3 {level = 3 : i64, scale = 52 : i64} : (!encoder, tensor<4096xf64>) -> !pt
    %ct_37 = cheddar.add_plain %ctx, %ct_35, %pt_36 : (!ctx, !ct, !pt) -> !ct
    %ct_38 = cheddar.rescale %ctx, %ct_37 : (!ctx, !ct) -> !ct
    %ct_39 = cheddar.mult %ctx, %ct_38, %ct_38 : (!ctx, !ct, !ct) -> !ct
    %evk_40 = cheddar.get_mult_key %ui : (!ui) -> !evk
    %ct_41 = cheddar.relinearize %ctx, %ct_39, %evk_40 : (!ctx, !ct, !evk) -> !ct
    %evk_map_42 = cheddar.get_evk_map %ui : (!ui) -> !evk_map
    %ct_43 = cheddar.linear_transform %ctx, %ct_41, %evk_map_42, %arg4 {diagonal_indices = array<i32: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 4087, 4088, 4089, 4090, 4091, 4092, 4093, 4094, 4095>, level = 2 : i64, logBabyStepGiantStepRatio = 1 : i64} : (!ctx, !ct, !evk_map, tensor<137x4096xf64>) -> !ct
    %pt_44 = cheddar.encode %encoder, %arg5 {level = 1 : i64, scale = 52 : i64} : (!encoder, tensor<4096xf64>) -> !pt
    %ct_45 = cheddar.add_plain %ctx, %ct_43, %pt_44 : (!ctx, !ct, !pt) -> !ct
    %ct_46 = cheddar.rescale %ctx, %ct_45 : (!ctx, !ct) -> !ct
    return %ct_46 : !ct
  }
  func.func @mlp__encrypt__arg0(%ctx: !ctx, %encoder: !encoder, %ui: !ui, %arg0: tensor<4096xf64>, %ui_0: !ui) -> !ct attributes {client.enc_func = {func_name = "mlp", index = 0 : i64}} {
    %pt = cheddar.encode %encoder, %arg0 {level = 5 : i64, scale = 26 : i64} : (!encoder, tensor<4096xf64>) -> !pt
    %ct = cheddar.encrypt %ui, %pt : (!ui, !pt) -> !ct
    return %ct : !ct
  }
  func.func @mlp__decrypt__result0(%ctx: !ctx, %encoder: !encoder, %ui: !ui, %ct: !ct, %ui_0: !ui) -> tensor<4096xf64> attributes {client.dec_func = {func_name = "mlp", index = 0 : i64}} {
    %pt = cheddar.decrypt %ui, %ct : (!ui, !ct) -> !pt
    %0 = cheddar.decode %encoder, %pt : (!encoder, !pt) -> tensor<4096xf64>
    return %0 : tensor<4096xf64>
  }
}
