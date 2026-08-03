// RUN: heir-opt --lattigo-configure-crypto-context %s | FileCheck %s

// Two bootstrap-evaluator args tagged with distinct logSlots must drive the
// configure func to generate two separate bootstrapping parameter literals,
// param-from-literal ops, evaluator-key sets, and bootstrapping evaluators.

!bootstrapping_evaluator = !lattigo.ckks.bootstrapping_evaluator
!ct = !lattigo.rlwe.ciphertext
!encoder = !lattigo.ckks.encoder
!evaluator = !lattigo.ckks.evaluator
!param = !lattigo.ckks.parameter
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 17, Q = [1106058412451299513, 1056763241666817029, 957769724367225479, 919081519653443687, 1030837924888066153, 1084354410096143723, 1135846243351935917, 1087115004561311021, 997960547764032911, 892538949448853293, 1002528331340998513, 1100798419621231379, 981696679688787961, 1061922508412786269], P = [1152921504606846976], logDefaultScale = 60>, scheme.ckks} {
  func.func @bootstrap(
      %bt1: !bootstrapping_evaluator {lattigo.bootstrap_log_slots = 1 : i64},
      %bt2: !bootstrapping_evaluator {lattigo.bootstrap_log_slots = 2 : i64},
      %evaluator: !evaluator, %param: !param, %encoder: !encoder,
      %ct: !ct) -> !ct {
    %ct_0 = lattigo.ckks.bootstrap %bt1, %ct : (!bootstrapping_evaluator, !ct) -> !ct
    %ct_1 = lattigo.ckks.bootstrap %bt2, %ct_0 : (!bootstrapping_evaluator, !ct) -> !ct
    return %ct_1 : !ct
  }
}

// CHECK-DAG: ![[btEvalType:.*]] = !lattigo.ckks.bootstrapping_evaluator
// CHECK-DAG: ![[evalType:.*]] = !lattigo.ckks.evaluator

// CHECK: @bootstrap__configure
// Configure returns two bootstrap evaluators, then the regular evaluator.
// CHECK-SAME: -> (![[btEvalType]], ![[btEvalType]], ![[evalType]],

// CHECK: %[[param:.*]] = lattigo.ckks.new_parameters_from_literal
// CHECK: %[[btParams1:.*]] = lattigo.ckks.new_bootstrapping_parameters_from_literal %[[param]] {btParamsLiteral = #lattigo.ckks.bootstrapping_parameters_literal<logN = 17, logSlots = 1>}
// CHECK: %[[btKeys1:.*]] = lattigo.ckks.gen_evaluation_keys_bootstrapping %[[btParams1]]
// CHECK: %[[btEval1:.*]] = lattigo.ckks.new_bootstrapping_evaluator %[[btParams1]], %[[btKeys1]]
// CHECK: %[[btParams2:.*]] = lattigo.ckks.new_bootstrapping_parameters_from_literal %[[param]] {btParamsLiteral = #lattigo.ckks.bootstrapping_parameters_literal<logN = 17, logSlots = 2>}
// CHECK: %[[btKeys2:.*]] = lattigo.ckks.gen_evaluation_keys_bootstrapping %[[btParams2]]
// CHECK: %[[btEval2:.*]] = lattigo.ckks.new_bootstrapping_evaluator %[[btParams2]], %[[btKeys2]]
