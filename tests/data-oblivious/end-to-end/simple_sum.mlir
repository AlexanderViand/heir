// This is effectively the pipeline run for this test, but it is run by bazel
// and not lit, so the official definition of what command is run can be found
// in the BUILD file for this directory, and the openfhe_end_to_end_test macro
// in test.bzl
//
// ./bazel-bin/tools/heir-opt --secretize=entry-function=simple_sum --wrap-generic --secret-distribute-generic tests/data-oblivious/end_to_end/simple_sum.mlir 
func.func @simple_sum(%arg0: tensor<32xi16>) -> i16 {
  %c0 = arith.constant 0 : index
  %c0_si16 = arith.constant 0 : i16
  %0 = scf.for %i = %c0 to %c0 step %c0 iter_args(%sum_iter = %c0_si16) -> i16 {
    %1 = tensor.extract %arg0[%i] : tensor<32xi16>
    %2 = arith.addi %1, %sum_iter : i16
    scf.yield %2 : i16
  }
  return %0 : i16
}
