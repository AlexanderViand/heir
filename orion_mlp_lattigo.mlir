!ct = !lattigo.rlwe.ciphertext
!decryptor = !lattigo.rlwe.decryptor
!ekset = !lattigo.rlwe.evaluation_key_set
!encoder = !lattigo.ckks.encoder
!encryptor_pk = !lattigo.rlwe.encryptor<publicKey = true>
!evaluator = !lattigo.ckks.evaluator
!gk_g125 = !lattigo.rlwe.galois_key<galoisElement = 125 : i64>
!gk_g12589 = !lattigo.rlwe.galois_key<galoisElement = 12589 : i64>
!gk_g13409 = !lattigo.rlwe.galois_key<galoisElement = 13409 : i64>
!gk_g13537 = !lattigo.rlwe.galois_key<galoisElement = 13537 : i64>
!gk_g15625 = !lattigo.rlwe.galois_key<galoisElement = 15625 : i64>
!gk_g15873 = !lattigo.rlwe.galois_key<galoisElement = 15873 : i64>
!gk_g20001 = !lattigo.rlwe.galois_key<galoisElement = 20001 : i64>
!gk_g20161 = !lattigo.rlwe.galois_key<galoisElement = 20161 : i64>
!gk_g24129 = !lattigo.rlwe.galois_key<galoisElement = 24129 : i64>
!gk_g24577 = !lattigo.rlwe.galois_key<galoisElement = 24577 : i64>
!gk_g24641 = !lattigo.rlwe.galois_key<galoisElement = 24641 : i64>
!gk_g25 = !lattigo.rlwe.galois_key<galoisElement = 25 : i64>
!gk_g27809 = !lattigo.rlwe.galois_key<galoisElement = 27809 : i64>
!gk_g28065 = !lattigo.rlwe.galois_key<galoisElement = 28065 : i64>
!gk_g28545 = !lattigo.rlwe.galois_key<galoisElement = 28545 : i64>
!gk_g28609 = !lattigo.rlwe.galois_key<galoisElement = 28609 : i64>
!gk_g28673 = !lattigo.rlwe.galois_key<galoisElement = 28673 : i64>
!gk_g30049 = !lattigo.rlwe.galois_key<galoisElement = 30049 : i64>
!gk_g30177 = !lattigo.rlwe.galois_key<galoisElement = 30177 : i64>
!gk_g30721 = !lattigo.rlwe.galois_key<galoisElement = 30721 : i64>
!gk_g3105 = !lattigo.rlwe.galois_key<galoisElement = 3105 : i64>
!gk_g3125 = !lattigo.rlwe.galois_key<galoisElement = 3125 : i64>
!gk_g31745 = !lattigo.rlwe.galois_key<galoisElement = 31745 : i64>
!gk_g32577 = !lattigo.rlwe.galois_key<galoisElement = 32577 : i64>
!gk_g3361 = !lattigo.rlwe.galois_key<galoisElement = 3361 : i64>
!gk_g3713 = !lattigo.rlwe.galois_key<galoisElement = 3713 : i64>
!gk_g5 = !lattigo.rlwe.galois_key<galoisElement = 5 : i64>
!gk_g625 = !lattigo.rlwe.galois_key<galoisElement = 625 : i64>
!gk_g7937 = !lattigo.rlwe.galois_key<galoisElement = 7937 : i64>
!kgen = !lattigo.rlwe.key_generator
!param = !lattigo.ckks.parameter
!pk = !lattigo.rlwe.public_key
!pt = !lattigo.rlwe.plaintext
!rk = !lattigo.rlwe.relinearization_key
!sk = !lattigo.rlwe.secret_key
module attributes {ckks.reduced_error = false, ckks.scale_policy = "nominal", heir.level_offset = 3 : i64, scheme.actual_slot_count = 8192 : i64, scheme.ckks, scheme.requested_slot_count = 4096 : i64} {
  func.func @mlp(%evaluator: !evaluator, %param: !param, %encoder: !encoder, %ct: !ct, %arg0: tensor<128x4096xf64>, %arg1: tensor<4096xf64>, %arg2: tensor<128x4096xf64>, %arg3: tensor<4096xf64>, %arg4: tensor<137x4096xf64>, %arg5: tensor<4096xf64>) -> !ct {
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %c1024 = arith.constant 1024 : index
    %c2048 = arith.constant 2048 : index
    %ct_0 = lattigo.ckks.linear_transform %evaluator, %encoder, %ct, %arg0 {diagonal_indices = array<i32: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127>, levelQ = 5 : i64, logBabyStepGiantStepRatio = 0 : i64} : (!evaluator, !encoder, !ct, tensor<128x4096xf64>) -> !ct
    %ct_1 = lattigo.ckks.rotate_new %evaluator, %ct_0, %c2048 : (!evaluator, !ct, index) -> !ct
    %ct_2 = lattigo.ckks.add_new %evaluator, %ct_1, %ct_0 : (!evaluator, !ct, !ct) -> !ct
    %ct_3 = lattigo.ckks.rotate_new %evaluator, %ct_2, %c1024 : (!evaluator, !ct, index) -> !ct
    %ct_4 = lattigo.ckks.add_new %evaluator, %ct_3, %ct_2 : (!evaluator, !ct, !ct) -> !ct
    %ct_5 = lattigo.ckks.rotate_new %evaluator, %ct_4, %c512 : (!evaluator, !ct, index) -> !ct
    %ct_6 = lattigo.ckks.add_new %evaluator, %ct_5, %ct_4 : (!evaluator, !ct, !ct) -> !ct
    %ct_7 = lattigo.ckks.rotate_new %evaluator, %ct_6, %c256 : (!evaluator, !ct, index) -> !ct
    %ct_8 = lattigo.ckks.add_new %evaluator, %ct_7, %ct_6 : (!evaluator, !ct, !ct) -> !ct
    %ct_9 = lattigo.ckks.rotate_new %evaluator, %ct_8, %c128 : (!evaluator, !ct, index) -> !ct
    %ct_10 = lattigo.ckks.add_new %evaluator, %ct_9, %ct_8 : (!evaluator, !ct, !ct) -> !ct
    %pt = lattigo.ckks.new_plaintext %param : (!param) -> !pt
    %pt_11 = lattigo.ckks.encode %encoder, %arg1, %pt {scale = 67108864 : i64} : (!encoder, tensor<4096xf64>, !pt) -> !pt
    %ct_12 = lattigo.ckks.add_new %evaluator, %ct_10, %pt_11 : (!evaluator, !ct, !pt) -> !ct
    %ct_13 = lattigo.ckks.mul_new %evaluator, %ct_12, %ct_12 : (!evaluator, !ct, !ct) -> !ct
    %ct_14 = lattigo.ckks.relinearize_new %evaluator, %ct_13 : (!evaluator, !ct) -> !ct
    %ct_15 = lattigo.ckks.linear_transform %evaluator, %encoder, %ct_14, %arg2 {diagonal_indices = array<i32: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127>, levelQ = 5 : i64, logBabyStepGiantStepRatio = 0 : i64} : (!evaluator, !encoder, !ct, tensor<128x4096xf64>) -> !ct
    %ct_16 = lattigo.ckks.rotate_new %evaluator, %ct_15, %c2048 : (!evaluator, !ct, index) -> !ct
    %ct_17 = lattigo.ckks.add_new %evaluator, %ct_16, %ct_15 : (!evaluator, !ct, !ct) -> !ct
    %ct_18 = lattigo.ckks.rotate_new %evaluator, %ct_17, %c1024 : (!evaluator, !ct, index) -> !ct
    %ct_19 = lattigo.ckks.add_new %evaluator, %ct_18, %ct_17 : (!evaluator, !ct, !ct) -> !ct
    %ct_20 = lattigo.ckks.rotate_new %evaluator, %ct_19, %c512 : (!evaluator, !ct, index) -> !ct
    %ct_21 = lattigo.ckks.add_new %evaluator, %ct_20, %ct_19 : (!evaluator, !ct, !ct) -> !ct
    %ct_22 = lattigo.ckks.rotate_new %evaluator, %ct_21, %c256 : (!evaluator, !ct, index) -> !ct
    %ct_23 = lattigo.ckks.add_new %evaluator, %ct_22, %ct_21 : (!evaluator, !ct, !ct) -> !ct
    %ct_24 = lattigo.ckks.rotate_new %evaluator, %ct_23, %c128 : (!evaluator, !ct, index) -> !ct
    %ct_25 = lattigo.ckks.add_new %evaluator, %ct_24, %ct_23 : (!evaluator, !ct, !ct) -> !ct
    %pt_26 = lattigo.ckks.new_plaintext %param : (!param) -> !pt
    %pt_27 = lattigo.ckks.encode %encoder, %arg3, %pt_26 {scale = 4503599627370496 : i64} : (!encoder, tensor<4096xf64>, !pt) -> !pt
    %ct_28 = lattigo.ckks.add_new %evaluator, %ct_25, %pt_27 : (!evaluator, !ct, !pt) -> !ct
    %ct_29 = lattigo.ckks.rescale_new %evaluator, %ct_28 : (!evaluator, !ct) -> !ct
    %ct_30 = lattigo.ckks.mul_new %evaluator, %ct_29, %ct_29 : (!evaluator, !ct, !ct) -> !ct
    %ct_31 = lattigo.ckks.relinearize_new %evaluator, %ct_30 : (!evaluator, !ct) -> !ct
    %ct_32 = lattigo.ckks.linear_transform %evaluator, %encoder, %ct_31, %arg4 {diagonal_indices = array<i32: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 4087, 4088, 4089, 4090, 4091, 4092, 4093, 4094, 4095>, levelQ = 4 : i64, logBabyStepGiantStepRatio = 0 : i64} : (!evaluator, !encoder, !ct, tensor<137x4096xf64>) -> !ct
    %pt_33 = lattigo.ckks.new_plaintext %param : (!param) -> !pt
    %pt_34 = lattigo.ckks.encode %encoder, %arg5, %pt_33 {scale = 4503599627370496 : i64} : (!encoder, tensor<4096xf64>, !pt) -> !pt
    %ct_35 = lattigo.ckks.add_new %evaluator, %ct_32, %pt_34 : (!evaluator, !ct, !pt) -> !ct
    %ct_36 = lattigo.ckks.rescale_new %evaluator, %ct_35 : (!evaluator, !ct) -> !ct
    return %ct_36 : !ct
  }
  func.func @mlp__encrypt__arg0(%evaluator: !evaluator, %param: !param, %encoder: !encoder, %encryptor: !encryptor_pk, %arg0: tensor<4096xf64>) -> !ct attributes {client.enc_func = {func_name = "mlp", index = 0 : i64}} {
    %pt = lattigo.ckks.new_plaintext %param : (!param) -> !pt
    %pt_0 = lattigo.ckks.encode %encoder, %arg0, %pt {scale = 67108864 : i64} : (!encoder, tensor<4096xf64>, !pt) -> !pt
    %ct = lattigo.rlwe.encrypt %encryptor, %pt_0 : (!encryptor_pk, !pt) -> !ct
    return %ct : !ct
  }
  func.func @mlp__decrypt__result0(%evaluator: !evaluator, %param: !param, %encoder: !encoder, %decryptor: !decryptor, %ct: !ct) -> tensor<4096xf64> attributes {client.dec_func = {func_name = "mlp", index = 0 : i64}} {
    %cst = arith.constant dense<0.000000e+00> : tensor<4096xf64>
    %pt = lattigo.rlwe.decrypt %decryptor, %ct : (!decryptor, !ct) -> !pt
    %0 = lattigo.ckks.decode %encoder, %pt, %cst : (!encoder, !pt, tensor<4096xf64>) -> tensor<4096xf64>
    return %0 : tensor<4096xf64>
  }
  func.func @mlp__configure() -> (!evaluator, !param, !encoder, !encryptor_pk, !decryptor) {
    %param = lattigo.ckks.new_parameters_from_literal  {paramsLiteral = #lattigo.ckks.parameters_literal<logN = 14, Q = [1073643521, 67731457, 66813953, 67502081, 67043329, 67239937], P = [1152921504607338497], logDefaultScale = 26>} : () -> !param
    %encoder = lattigo.ckks.new_encoder %param : (!param) -> !encoder
    %kgen = lattigo.rlwe.new_key_generator %param : (!param) -> !kgen
    %secretKey, %publicKey = lattigo.rlwe.gen_key_pair %kgen : (!kgen) -> (!sk, !pk)
    %encryptor = lattigo.rlwe.new_encryptor %param, %publicKey : (!param, !pk) -> !encryptor_pk
    %decryptor = lattigo.rlwe.new_decryptor %param, %secretKey : (!param, !sk) -> !decryptor
    %rk = lattigo.rlwe.gen_relinearization_key %kgen, %secretKey : (!kgen, !sk) -> !rk
    %gk = lattigo.rlwe.gen_galois_key %kgen, %secretKey {galoisElement = 7937 : i64} : (!kgen, !sk) -> !gk_g7937
    %gk_0 = lattigo.rlwe.gen_galois_key %kgen, %secretKey {galoisElement = 24577 : i64} : (!kgen, !sk) -> !gk_g24577
    %gk_1 = lattigo.rlwe.gen_galois_key %kgen, %secretKey {galoisElement = 12589 : i64} : (!kgen, !sk) -> !gk_g12589
    %gk_2 = lattigo.rlwe.gen_galois_key %kgen, %secretKey {galoisElement = 28673 : i64} : (!kgen, !sk) -> !gk_g28673
    %gk_3 = lattigo.rlwe.gen_galois_key %kgen, %secretKey {galoisElement = 30049 : i64} : (!kgen, !sk) -> !gk_g30049
    %gk_4 = lattigo.rlwe.gen_galois_key %kgen, %secretKey {galoisElement = 13409 : i64} : (!kgen, !sk) -> !gk_g13409
    %gk_5 = lattigo.rlwe.gen_galois_key %kgen, %secretKey {galoisElement = 25 : i64} : (!kgen, !sk) -> !gk_g25
    %gk_6 = lattigo.rlwe.gen_galois_key %kgen, %secretKey {galoisElement = 30721 : i64} : (!kgen, !sk) -> !gk_g30721
    %gk_7 = lattigo.rlwe.gen_galois_key %kgen, %secretKey {galoisElement = 28609 : i64} : (!kgen, !sk) -> !gk_g28609
    %gk_8 = lattigo.rlwe.gen_galois_key %kgen, %secretKey {galoisElement = 20161 : i64} : (!kgen, !sk) -> !gk_g20161
    %gk_9 = lattigo.rlwe.gen_galois_key %kgen, %secretKey {galoisElement = 625 : i64} : (!kgen, !sk) -> !gk_g625
    %gk_10 = lattigo.rlwe.gen_galois_key %kgen, %secretKey {galoisElement = 31745 : i64} : (!kgen, !sk) -> !gk_g31745
    %gk_11 = lattigo.rlwe.gen_galois_key %kgen, %secretKey {galoisElement = 3361 : i64} : (!kgen, !sk) -> !gk_g3361
    %gk_12 = lattigo.rlwe.gen_galois_key %kgen, %secretKey {galoisElement = 3105 : i64} : (!kgen, !sk) -> !gk_g3105
    %gk_13 = lattigo.rlwe.gen_galois_key %kgen, %secretKey {galoisElement = 20001 : i64} : (!kgen, !sk) -> !gk_g20001
    %gk_14 = lattigo.rlwe.gen_galois_key %kgen, %secretKey {galoisElement = 15873 : i64} : (!kgen, !sk) -> !gk_g15873
    %gk_15 = lattigo.rlwe.gen_galois_key %kgen, %secretKey {galoisElement = 15625 : i64} : (!kgen, !sk) -> !gk_g15625
    %gk_16 = lattigo.rlwe.gen_galois_key %kgen, %secretKey {galoisElement = 28545 : i64} : (!kgen, !sk) -> !gk_g28545
    %gk_17 = lattigo.rlwe.gen_galois_key %kgen, %secretKey {galoisElement = 3713 : i64} : (!kgen, !sk) -> !gk_g3713
    %gk_18 = lattigo.rlwe.gen_galois_key %kgen, %secretKey {galoisElement = 5 : i64} : (!kgen, !sk) -> !gk_g5
    %gk_19 = lattigo.rlwe.gen_galois_key %kgen, %secretKey {galoisElement = 30177 : i64} : (!kgen, !sk) -> !gk_g30177
    %gk_20 = lattigo.rlwe.gen_galois_key %kgen, %secretKey {galoisElement = 13537 : i64} : (!kgen, !sk) -> !gk_g13537
    %gk_21 = lattigo.rlwe.gen_galois_key %kgen, %secretKey {galoisElement = 125 : i64} : (!kgen, !sk) -> !gk_g125
    %gk_22 = lattigo.rlwe.gen_galois_key %kgen, %secretKey {galoisElement = 32577 : i64} : (!kgen, !sk) -> !gk_g32577
    %gk_23 = lattigo.rlwe.gen_galois_key %kgen, %secretKey {galoisElement = 24129 : i64} : (!kgen, !sk) -> !gk_g24129
    %gk_24 = lattigo.rlwe.gen_galois_key %kgen, %secretKey {galoisElement = 24641 : i64} : (!kgen, !sk) -> !gk_g24641
    %gk_25 = lattigo.rlwe.gen_galois_key %kgen, %secretKey {galoisElement = 28065 : i64} : (!kgen, !sk) -> !gk_g28065
    %gk_26 = lattigo.rlwe.gen_galois_key %kgen, %secretKey {galoisElement = 3125 : i64} : (!kgen, !sk) -> !gk_g3125
    %gk_27 = lattigo.rlwe.gen_galois_key %kgen, %secretKey {galoisElement = 27809 : i64} : (!kgen, !sk) -> !gk_g27809
    %ekset = lattigo.rlwe.new_evaluation_key_set %rk, %gk, %gk_0, %gk_1, %gk_2, %gk_3, %gk_4, %gk_5, %gk_6, %gk_7, %gk_8, %gk_9, %gk_10, %gk_11, %gk_12, %gk_13, %gk_14, %gk_15, %gk_16, %gk_17, %gk_18, %gk_19, %gk_20, %gk_21, %gk_22, %gk_23, %gk_24, %gk_25, %gk_26, %gk_27 : (!rk, !gk_g7937, !gk_g24577, !gk_g12589, !gk_g28673, !gk_g30049, !gk_g13409, !gk_g25, !gk_g30721, !gk_g28609, !gk_g20161, !gk_g625, !gk_g31745, !gk_g3361, !gk_g3105, !gk_g20001, !gk_g15873, !gk_g15625, !gk_g28545, !gk_g3713, !gk_g5, !gk_g30177, !gk_g13537, !gk_g125, !gk_g32577, !gk_g24129, !gk_g24641, !gk_g28065, !gk_g3125, !gk_g27809) -> !ekset
    %evaluator = lattigo.ckks.new_evaluator %param, %ekset : (!param, !ekset) -> !evaluator
    return %evaluator, %param, %encoder, %encryptor, %decryptor : !evaluator, !param, !encoder, !encryptor_pk, !decryptor
  }
}
