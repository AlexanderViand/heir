// bazel run //tools:heir-opt -- --secretize=entry-function="simple_if" --wrap-generic --convert-if-to-select --canonicalize --cse $PWD/tests/data-oblivious/simple_if/simple_if.mlir
func.func @simple_if(%cond: i1, %input: i32) -> (i32){
    %result = scf.if %cond -> (i32) {
        %add = arith.addi %input, %input : i32
        scf.yield %add : i32
    } else{
        %mul = arith.muli %input, %input : i32
        scf.yield %mul : i32
    }
    return %result : i32
}

func.func @main() -> (i32){
    %cond = arith.constant 0 : i1
    %input = arith.constant 10 : i32
    %result = call @simple_if(%cond, %input) : (i1, i32) -> i32
    return %result : i32
}
