// RUN: heir-translate %s --emit-lattigo | FileCheck %s

// Two distinct bootstrapping parameter literals with different sparse logSlots
// must emit two `bootstrapping.NewParametersFromLiteral` calls, each with the
// expected `LogSlots: utils.Pointy(N)`.

!params = !lattigo.ckks.parameter
!bt_params = !lattigo.ckks.bootstrapping_parameter
!bt_keys = !lattigo.ckks.bootstrapping_eval_keys
!bt_eval = !lattigo.ckks.bootstrapping_evaluator
!sk = !lattigo.rlwe.secret_key

#paramsLiteral = #lattigo.ckks.parameters_literal<
    logN = 14,
    logQ = [55, 45, 45],
    logP = [61],
    logDefaultScale = 45
>

module attributes {scheme.ckks} {
  // CHECK: func make_bootstrappers
  // CHECK: bootstrapping.NewParametersFromLiteral
  // CHECK: LogN: utils.Pointy(14)
  // CHECK: LogSlots: utils.Pointy(1)
  // CHECK: bootstrapping.NewParametersFromLiteral
  // CHECK: LogN: utils.Pointy(14)
  // CHECK: LogSlots: utils.Pointy(2)
  func.func @make_bootstrappers(%sk: !sk) -> (!bt_eval, !bt_eval) {
    %params = lattigo.ckks.new_parameters_from_literal {paramsLiteral = #paramsLiteral} : () -> !params
    %bt_params_1 = lattigo.ckks.new_bootstrapping_parameters_from_literal %params {btParamsLiteral = #lattigo.ckks.bootstrapping_parameters_literal<logN = 14, logSlots = 1>} : (!params) -> !bt_params
    %bt_keys_1 = lattigo.ckks.gen_evaluation_keys_bootstrapping %bt_params_1, %sk : (!bt_params, !sk) -> !bt_keys
    %bt_eval_1 = lattigo.ckks.new_bootstrapping_evaluator %bt_params_1, %bt_keys_1 : (!bt_params, !bt_keys) -> !bt_eval
    %bt_params_2 = lattigo.ckks.new_bootstrapping_parameters_from_literal %params {btParamsLiteral = #lattigo.ckks.bootstrapping_parameters_literal<logN = 14, logSlots = 2>} : (!params) -> !bt_params
    %bt_keys_2 = lattigo.ckks.gen_evaluation_keys_bootstrapping %bt_params_2, %sk : (!bt_params, !sk) -> !bt_keys
    %bt_eval_2 = lattigo.ckks.new_bootstrapping_evaluator %bt_params_2, %bt_keys_2 : (!bt_params, !bt_keys) -> !bt_eval
    return %bt_eval_1, %bt_eval_2 : !bt_eval, !bt_eval
  }
}
