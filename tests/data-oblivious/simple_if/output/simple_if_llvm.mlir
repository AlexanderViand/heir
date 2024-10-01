module {
  llvm.func @simple_if(%arg0: i1, %arg1: i32) -> i32 {
    llvm.cond_br %arg0, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %0 = llvm.add %arg1, %arg1 : i32
    llvm.br ^bb3(%0 : i32)
  ^bb2:  // pred: ^bb0
    %1 = llvm.mul %arg1, %arg1 : i32
    llvm.br ^bb3(%1 : i32)
  ^bb3(%2: i32):  // 2 preds: ^bb1, ^bb2
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    llvm.return %2 : i32
  }
  llvm.func @main() -> i32 {
    %0 = llvm.mlir.constant(false) : i1
    %1 = llvm.mlir.constant(10 : i32) : i32
    %2 = llvm.call @simple_if(%0, %1) : (i1, i32) -> i32
    llvm.return %2 : i32
  }
}
