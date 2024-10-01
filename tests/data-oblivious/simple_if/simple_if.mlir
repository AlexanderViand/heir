// bazel run //tools:heir-opt -- --secretize=entry-function="simple_if" --wrap-generic --convert-if-to-select --canonicalize --cse $PWD/tests/data-oblivious/simple_if/simple_if.mlir
func.func @simple_if(%cond: i1, %input: i16) -> (i16){
    %result = scf.if %cond -> (i16) {
        %add = arith.addi %input, %input : i16
        scf.yield %add : i16
    } else{
        %mul = arith.muli %input, %input : i16
        scf.yield %mul : i16
    }
    return %result : i16
}
