// RUN: heir-translate %s --emit-lattigo | FileCheck %s

!bt_eval = !lattigo.ckks.bootstrapping_evaluator
!ct = !lattigo.rlwe.ciphertext

module attributes {scheme.ckks} {
  // CHECK: func Bootstrap_real
  // CHECK: [[CT:[^. ]+]].Scale = [[CT]].Scale.Mul(rlwe.NewScale({{2.*}}))
  // CHECK: [[OUT:[^, ]+]], [[ERR:[^ ]+]] := [[EVAL:[^. ]+]].Bootstrap([[CT]])
  // CHECK: [[CT]].Scale = [[CT]].Scale.Div(rlwe.NewScale({{2.*}}))
  // CHECK: [[CONJ:[^, ]+]], [[ERR]] := [[EVAL]].ConjugateNew([[OUT]])
  // CHECK: [[OUT]], [[ERR]] = [[EVAL]].AddNew([[OUT]], [[CONJ]])
  func.func @bootstrap_real(%evaluator: !bt_eval, %ct: !ct) -> !ct {
    %result = lattigo.ckks.bootstrap %evaluator, %ct {inputScaleMultiplier = 2.0 : f64, realify = true} : (!bt_eval, !ct) -> !ct
    return %result : !ct
  }
}
