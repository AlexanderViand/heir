// bazel run //tools:heir-opt -- --convert-secret-for-to-static-for --convert-if-to-select $PWD/tests/data-oblivious/partial_sum/partial_sum.mlir
func.func @partial_sum_test(%secretInput :!secret.secret<tensor<16xi16>>, %secretIndex: !secret.secret<index>, %secretThreshold : !secret.secret<i16>) -> (!secret.secret<i16>, !secret.secret<i16>) {
    %start = arith.constant 0 : index
    %step = arith.constant 1 : index
    %c0 = arith.constant 0 : i16

    %0, %1 = secret.generic ins(%secretInput, %secretIndex, %secretThreshold : !secret.secret<tensor<16xi16>>, !secret.secret<index>, !secret.secret<i16>) {
    ^bb0(%input: tensor<16xi16>, %index: index, %threshold: i16):
        %2, %3 = scf.for %i = %start to %index step %step iter_args(%arg1 = %c0, %arg2 = %c0) -> (i16, i16) {
            %cond = arith.cmpi slt, %arg1, %threshold : i16
            %extracted = tensor.extract %input[%i] : tensor<16xi16>
            %sum1, %sum2 = scf.if %cond -> (i16, i16) {
                %sum = arith.addi %arg1, %extracted : i16
                scf.yield %sum, %arg2 : i16, i16
            } else {
                %sum = arith.addi %arg2, %extracted : i16
                scf.yield %arg1, %sum : i16, i16
            }
            scf.yield %sum1, %sum2 : i16, i16
        } {lower = 0, upper = 16}
        secret.yield %2, %3 : i16, i16
    }-> (!secret.secret<i16>, !secret.secret<i16>)
   return %0, %1 : !secret.secret<i16>, !secret.secret<i16>
}
