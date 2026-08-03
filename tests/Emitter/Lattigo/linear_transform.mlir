// RUN: heir-translate %s --emit-lattigo | FileCheck %s

// CHECK: lintrans.Diagonals
// CHECK: lintrans.Parameters
// CHECK: lintrans.NewTransformation
// CHECK: lintrans.Encode
// CHECK: lintrans.NewEvaluator
// CHECK: EvaluateNew
// CHECK: [[PT:.*]].Scale = [[CT:.*]].Scale

!ct = !lattigo.rlwe.ciphertext
!pt = !lattigo.rlwe.plaintext
!encoder = !lattigo.ckks.encoder
!evaluator = !lattigo.ckks.evaluator
!param = !lattigo.ckks.parameter
module attributes {scheme.ckks} {
  func.func @linear_transform(%evaluator: !evaluator, %param: !param, %encoder: !encoder, %ct: !ct, %arg0: tensor<2x4096xf64>) -> !ct {
    %ct_0 = lattigo.ckks.linear_transform %evaluator, %encoder, %ct, %arg0 {levelQ = 5 : i32, logBabyStepGiantStepRatio = 2 : i64, diagonal_indices = array<i32: 0, 1>} : (!evaluator, !encoder, !ct, tensor<2x4096xf64>) -> !ct
    %ct_1 = lattigo.ckks.rotate_new %evaluator, %ct_0 {static_shift = 2048 : i32} : (!evaluator, !ct) -> !ct
    %ct_2 = lattigo.ckks.add_new %evaluator, %ct_1, %ct_0 : (!evaluator, !ct, !ct) -> !ct
    return %ct_2 : !ct
  }

  func.func @runtime_scale_encode(%param: !param, %encoder: !encoder, %ct: !ct, %value: tensor<4xf32>) -> !pt {
    %raw = lattigo.ckks.new_plaintext %param : (!param) -> !pt
    %encoded = lattigo.ckks.encode %encoder, %value, %raw, %ct {scale = 45 : i64} : (!encoder, tensor<4xf32>, !pt, !ct) -> !pt
    return %encoded : !pt
  }
}
