module {
  llvm.func @free(!llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @set_intersect(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: i64, %arg8: i64, %arg9: i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.mlir.constant(true) : i1
    %2 = llvm.mlir.constant(false) : i1
    %3 = llvm.mlir.zero : !llvm.ptr
    %4 = llvm.mlir.constant(64 : index) : i64
    %5 = llvm.mlir.constant(1 : index) : i64
    %6 = llvm.mlir.constant(16 : index) : i64
    %7 = llvm.mlir.constant(0 : index) : i64
    %8 = llvm.getelementptr %3[16] : (!llvm.ptr) -> !llvm.ptr, i1
    %9 = llvm.ptrtoint %8 : !llvm.ptr to i64
    %10 = llvm.add %9, %4 : i64
    %11 = llvm.call @malloc(%10) : (i64) -> !llvm.ptr
    %12 = llvm.ptrtoint %11 : !llvm.ptr to i64
    %13 = llvm.sub %4, %5 : i64
    %14 = llvm.add %12, %13 : i64
    %15 = llvm.urem %14, %4  : i64
    %16 = llvm.sub %14, %15 : i64
    %17 = llvm.inttoptr %16 : i64 to !llvm.ptr
    llvm.br ^bb1(%7 : i64)
  ^bb1(%18: i64):  // 2 preds: ^bb0, ^bb5
    %19 = llvm.icmp "slt" %18, %6 : i64
    llvm.cond_br %19, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%7, %2 : i64, i1)
  ^bb3(%20: i64, %21: i1):  // 2 preds: ^bb2, ^bb4
    %22 = llvm.icmp "slt" %20, %6 : i64
    llvm.cond_br %22, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %23 = llvm.getelementptr %arg1[%arg2] : (!llvm.ptr, i64) -> !llvm.ptr, i16
    %24 = llvm.mul %18, %arg4 : i64
    %25 = llvm.getelementptr %23[%24] : (!llvm.ptr, i64) -> !llvm.ptr, i16
    %26 = llvm.load %25 : !llvm.ptr -> i16
    %27 = llvm.getelementptr %arg6[%arg7] : (!llvm.ptr, i64) -> !llvm.ptr, i16
    %28 = llvm.mul %20, %arg9 : i64
    %29 = llvm.getelementptr %27[%28] : (!llvm.ptr, i64) -> !llvm.ptr, i16
    %30 = llvm.load %29 : !llvm.ptr -> i16
    %31 = llvm.icmp "eq" %26, %30 : i16
    %32 = llvm.select %31, %1, %21 : i1, i1
    %33 = llvm.add %20, %5 : i64
    llvm.br ^bb3(%33, %32 : i64, i1)
  ^bb5:  // pred: ^bb3
    %34 = llvm.getelementptr %17[%18] : (!llvm.ptr, i64) -> !llvm.ptr, i1
    llvm.store %21, %34 : i1, !llvm.ptr
    %35 = llvm.add %18, %5 : i64
    llvm.br ^bb1(%35 : i64)
  ^bb6:  // pred: ^bb1
    %36 = llvm.call @malloc(%9) : (i64) -> !llvm.ptr
    %37 = llvm.insertvalue %36, %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %38 = llvm.insertvalue %36, %37[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %39 = llvm.insertvalue %7, %38[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %40 = llvm.insertvalue %6, %39[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %41 = llvm.insertvalue %5, %40[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %42 = llvm.mul %6, %5 : i64
    %43 = llvm.getelementptr %3[1] : (!llvm.ptr) -> !llvm.ptr, i1
    %44 = llvm.ptrtoint %43 : !llvm.ptr to i64
    %45 = llvm.mul %42, %44 : i64
    "llvm.intr.memcpy"(%36, %17, %45) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %46 = llvm.ptrtoint %17 : !llvm.ptr to i64
    %47 = llvm.ptrtoint %36 : !llvm.ptr to i64
    %48 = llvm.icmp "ne" %46, %47 : i64
    llvm.cond_br %48, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    llvm.call @free(%11) : (!llvm.ptr) -> ()
    llvm.br ^bb8
  ^bb8:  // 2 preds: ^bb6, ^bb7
    llvm.return %41 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  }
  llvm.func @main() -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.constant(2 : i16) : i16
    %2 = llvm.mlir.constant(64 : index) : i64
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.mlir.constant(16 : index) : i64
    %5 = llvm.mlir.constant(1 : index) : i64
    %6 = llvm.getelementptr %0[16] : (!llvm.ptr) -> !llvm.ptr, i16
    %7 = llvm.ptrtoint %6 : !llvm.ptr to i64
    %8 = llvm.add %7, %2 : i64
    %9 = llvm.call @malloc(%8) : (i64) -> !llvm.ptr
    %10 = llvm.ptrtoint %9 : !llvm.ptr to i64
    %11 = llvm.sub %2, %5 : i64
    %12 = llvm.add %10, %11 : i64
    %13 = llvm.urem %12, %2  : i64
    %14 = llvm.sub %12, %13 : i64
    %15 = llvm.inttoptr %14 : i64 to !llvm.ptr
    llvm.br ^bb1(%3 : i64)
  ^bb1(%16: i64):  // 2 preds: ^bb0, ^bb2
    %17 = llvm.icmp "slt" %16, %4 : i64
    llvm.cond_br %17, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %18 = llvm.getelementptr %15[%16] : (!llvm.ptr, i64) -> !llvm.ptr, i16
    llvm.store %1, %18 : i16, !llvm.ptr
    %19 = llvm.add %16, %5 : i64
    llvm.br ^bb1(%19 : i64)
  ^bb3:  // pred: ^bb1
    %20 = llvm.call @set_intersect(%9, %15, %3, %4, %5, %9, %15, %3, %4, %5) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %21 = llvm.ptrtoint %15 : !llvm.ptr to i64
    %22 = llvm.extractvalue %20[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %23 = llvm.ptrtoint %22 : !llvm.ptr to i64
    %24 = llvm.icmp "ne" %21, %23 : i64
    llvm.cond_br %24, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    llvm.call @free(%9) : (!llvm.ptr) -> ()
    llvm.br ^bb5
  ^bb5:  // 2 preds: ^bb3, ^bb4
    llvm.return %20 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  }
}
