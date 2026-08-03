// RUN: heir-opt --lattigo-configure-crypto-context %s | FileCheck %s

// When ckks.schemeParam carries `bootstrapLogP`, the configure pass must
// forward it onto the emitted lattigo.ckks.bootstrapping_parameters_literal so
// the downstream Go emission reproduces the frontend's bootstrap LogP rather
// than letting Lattigo pick a default.

!bootstrapping_evaluator = !lattigo.ckks.bootstrapping_evaluator
!ct = !lattigo.rlwe.ciphertext
!encoder = !lattigo.ckks.encoder
!evaluator = !lattigo.ckks.evaluator
!param = !lattigo.ckks.parameter
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 16, Q = [36028797018652673, 1099511922689, 1099512004609, 1099511693313, 1099512053761, 1099511627777, 1099512004609, 1099512725505, 1099513020417, 1099512184833, 1099513708545], P = [1152921504606584833, 1152921504593215489, 1152921504595705857], logDefaultScale = 40, bootstrapLogP = [61, 61, 61, 61, 61, 61, 61, 61]>, scheme.ckks} {
  func.func @bootstrap(%bootstrapping_evaluator: !bootstrapping_evaluator, %evaluator: !evaluator, %param: !param, %encoder: !encoder, %ct: !ct) -> !ct {
    %ct_0 = lattigo.ckks.bootstrap %bootstrapping_evaluator, %ct {inputScaleMultiplier = 2.0 : f64, realify = true} : (!bootstrapping_evaluator, !ct) -> !ct
    return %ct_0 : !ct
  }
}

// CHECK: lattigo.ckks.new_bootstrapping_parameters_from_literal
// CHECK-SAME: btParamsLiteral = #lattigo.ckks.bootstrapping_parameters_literal<logN = 16, logP = [61, 61, 61, 61, 61, 61, 61, 61]>
