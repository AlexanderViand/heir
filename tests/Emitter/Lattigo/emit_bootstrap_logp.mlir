// RUN: heir-translate %s --emit-lattigo | FileCheck %s

// Verify the emitter writes a `LogP: []int{...}` field on
// bootstrapping.ParametersLiteral when the IR carries `logP`.

!params = !lattigo.ckks.parameter
!bt_params = !lattigo.ckks.bootstrapping_parameter
!bt_keys = !lattigo.ckks.bootstrapping_eval_keys
!bt_eval = !lattigo.ckks.bootstrapping_evaluator
!sk = !lattigo.rlwe.secret_key

#paramsLiteral = #lattigo.ckks.parameters_literal<
    logN = 16,
    logQ = [55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40],
    logP = [61, 61, 61],
    logDefaultScale = 40
>

module attributes {scheme.ckks} {
  // CHECK: bootstrapping.NewParametersFromLiteral
  // CHECK: bootstrapping.ParametersLiteral{
  // CHECK: LogN: utils.Pointy(16)
  // CHECK: LogSlots: utils.Pointy(1)
  // CHECK: LogP: []int{61, 61, 61, 61, 61, 61, 61, 61}
  func.func @make_bootstrapper(%sk: !sk) -> !bt_eval {
    %params = lattigo.ckks.new_parameters_from_literal {paramsLiteral = #paramsLiteral} : () -> !params
    %bt_params = lattigo.ckks.new_bootstrapping_parameters_from_literal %params {btParamsLiteral = #lattigo.ckks.bootstrapping_parameters_literal<logN = 16, logSlots = 1, logP = [61, 61, 61, 61, 61, 61, 61, 61]>} : (!params) -> !bt_params
    %bt_keys = lattigo.ckks.gen_evaluation_keys_bootstrapping %bt_params, %sk : (!bt_params, !sk) -> !bt_keys
    %bt_eval = lattigo.ckks.new_bootstrapping_evaluator %bt_params, %bt_keys : (!bt_params, !bt_keys) -> !bt_eval
    return %bt_eval : !bt_eval
  }
}
