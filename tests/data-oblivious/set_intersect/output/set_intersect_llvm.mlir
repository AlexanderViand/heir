module {
  llvm.func @free(!llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.mlir.global private constant @__constant_16xi16_0(dense<[2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3]> : tensor<16xi16>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<16 x i16>
  llvm.mlir.global private constant @__constant_16xi16(dense<[2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4]> : tensor<16xi16>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<16 x i16>
  llvm.func private @printMemrefI32(%arg0: i64, %arg1: !llvm.ptr) attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i64, ptr)>
    %3 = llvm.insertvalue %arg1, %2[1] : !llvm.struct<(i64, ptr)>
    %4 = llvm.alloca %1 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %3, %4 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    llvm.call @_mlir_ciface_printMemrefI32(%4) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_printMemrefI32(!llvm.ptr) attributes {llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func @printTensor16(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64) attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.mlir.zero : !llvm.ptr
    %2 = llvm.mlir.constant(64 : index) : i64
    %3 = llvm.mlir.constant(1 : index) : i64
    %4 = llvm.mlir.constant(16 : index) : i64
    %5 = llvm.mlir.constant(0 : index) : i64
    %6 = llvm.getelementptr %1[16] : (!llvm.ptr) -> !llvm.ptr, i32
    %7 = llvm.ptrtoint %6 : !llvm.ptr to i64
    %8 = llvm.add %7, %2 : i64
    %9 = llvm.call @malloc(%8) : (i64) -> !llvm.ptr
    %10 = llvm.ptrtoint %9 : !llvm.ptr to i64
    %11 = llvm.sub %2, %3 : i64
    %12 = llvm.add %10, %11 : i64
    %13 = llvm.urem %12, %2  : i64
    %14 = llvm.sub %12, %13 : i64
    %15 = llvm.inttoptr %14 : i64 to !llvm.ptr
    %16 = llvm.insertvalue %9, %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %17 = llvm.insertvalue %15, %16[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %18 = llvm.insertvalue %5, %17[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %19 = llvm.insertvalue %4, %18[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %20 = llvm.insertvalue %3, %19[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    llvm.br ^bb1(%5 : i64)
  ^bb1(%21: i64):  // 2 preds: ^bb0, ^bb2
    %22 = llvm.icmp "slt" %21, %4 : i64
    llvm.cond_br %22, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %23 = llvm.getelementptr %arg1[%arg2] : (!llvm.ptr, i64) -> !llvm.ptr, i1
    %24 = llvm.mul %21, %arg4 : i64
    %25 = llvm.getelementptr %23[%24] : (!llvm.ptr, i64) -> !llvm.ptr, i1
    %26 = llvm.load %25 : !llvm.ptr -> i1
    %27 = llvm.zext %26 : i1 to i32
    %28 = llvm.getelementptr %15[%21] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %27, %28 : i32, !llvm.ptr
    %29 = llvm.add %21, %3 : i64
    llvm.br ^bb1(%29 : i64)
  ^bb3:  // pred: ^bb1
    %30 = llvm.alloca %3 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %20, %30 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    llvm.call @printMemrefI32(%3, %30) : (i64, !llvm.ptr) -> ()
    llvm.call @free(%9) : (!llvm.ptr) -> ()
    llvm.return
  }
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
  llvm.func @main() {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.constant(3 : i16) : i16
    %2 = llvm.mlir.constant(2 : i16) : i16
    %3 = llvm.mlir.constant(true) : i1
    %4 = llvm.mlir.addressof @__constant_16xi16 : !llvm.ptr
    %5 = llvm.mlir.addressof @__constant_16xi16_0 : !llvm.ptr
    %6 = llvm.mlir.constant(5 : index) : i64
    %7 = llvm.mlir.constant(64 : index) : i64
    %8 = llvm.mlir.constant(3735928559 : index) : i64
    %9 = llvm.mlir.constant(0 : index) : i64
    %10 = llvm.mlir.constant(1 : index) : i64
    %11 = llvm.mlir.constant(16 : index) : i64
    %12 = llvm.getelementptr %4[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i16>
    %13 = llvm.inttoptr %8 : i64 to !llvm.ptr
    %14 = llvm.getelementptr %5[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i16>
    %15 = llvm.getelementptr %0[16] : (!llvm.ptr) -> !llvm.ptr, i16
    %16 = llvm.ptrtoint %15 : !llvm.ptr to i64
    %17 = llvm.add %16, %7 : i64
    %18 = llvm.call @malloc(%17) : (i64) -> !llvm.ptr
    %19 = llvm.ptrtoint %18 : !llvm.ptr to i64
    %20 = llvm.sub %7, %10 : i64
    %21 = llvm.add %19, %20 : i64
    %22 = llvm.urem %21, %7  : i64
    %23 = llvm.sub %21, %22 : i64
    %24 = llvm.inttoptr %23 : i64 to !llvm.ptr
    llvm.br ^bb1(%9 : i64)
  ^bb1(%25: i64):  // 2 preds: ^bb0, ^bb2
    %26 = llvm.icmp "slt" %25, %11 : i64
    llvm.cond_br %26, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %27 = llvm.getelementptr %24[%25] : (!llvm.ptr, i64) -> !llvm.ptr, i16
    llvm.store %2, %27 : i16, !llvm.ptr
    %28 = llvm.add %25, %10 : i64
    llvm.br ^bb1(%28 : i64)
  ^bb3:  // pred: ^bb1
    %29 = llvm.call @malloc(%17) : (i64) -> !llvm.ptr
    %30 = llvm.ptrtoint %29 : !llvm.ptr to i64
    %31 = llvm.add %30, %20 : i64
    %32 = llvm.urem %31, %7  : i64
    %33 = llvm.sub %31, %32 : i64
    %34 = llvm.inttoptr %33 : i64 to !llvm.ptr
    llvm.br ^bb4(%9 : i64)
  ^bb4(%35: i64):  // 2 preds: ^bb3, ^bb5
    %36 = llvm.icmp "slt" %35, %11 : i64
    llvm.cond_br %36, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %37 = llvm.getelementptr %34[%35] : (!llvm.ptr, i64) -> !llvm.ptr, i16
    llvm.store %1, %37 : i16, !llvm.ptr
    %38 = llvm.add %35, %10 : i64
    llvm.br ^bb4(%38 : i64)
  ^bb6:  // pred: ^bb4
    %39 = llvm.call @set_intersect(%18, %24, %9, %11, %10, %18, %24, %9, %11, %10) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %40 = llvm.call @set_intersect(%18, %24, %9, %11, %10, %29, %34, %9, %11, %10) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %41 = llvm.call @set_intersect(%13, %14, %9, %11, %10, %13, %12, %9, %11, %10) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %42 = llvm.extractvalue %39[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %43 = llvm.extractvalue %39[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %44 = llvm.extractvalue %39[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %45 = llvm.extractvalue %39[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %46 = llvm.extractvalue %39[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @printTensor16(%42, %43, %44, %45, %46) : (!llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    %47 = llvm.extractvalue %40[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %48 = llvm.extractvalue %40[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %49 = llvm.extractvalue %40[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %50 = llvm.extractvalue %40[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %51 = llvm.extractvalue %40[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @printTensor16(%47, %48, %49, %50, %51) : (!llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    %52 = llvm.extractvalue %41[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %53 = llvm.extractvalue %41[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %54 = llvm.extractvalue %41[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %55 = llvm.extractvalue %41[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %56 = llvm.extractvalue %41[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @printTensor16(%52, %53, %54, %55, %56) : (!llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    %57 = llvm.getelementptr %0[5] : (!llvm.ptr) -> !llvm.ptr, i64
    %58 = llvm.ptrtoint %57 : !llvm.ptr to i64
    %59 = llvm.call @malloc(%58) : (i64) -> !llvm.ptr
    %60 = llvm.getelementptr %0[5] : (!llvm.ptr) -> !llvm.ptr, i1
    %61 = llvm.ptrtoint %60 : !llvm.ptr to i64
    %62 = llvm.call @malloc(%61) : (i64) -> !llvm.ptr
    %63 = llvm.getelementptr %0[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %64 = llvm.ptrtoint %63 : !llvm.ptr to i64
    %65 = llvm.call @malloc(%64) : (i64) -> !llvm.ptr
    %66 = llvm.ptrtoint %24 : !llvm.ptr to i64
    llvm.store %66, %59 : i64, !llvm.ptr
    %67 = llvm.ptrtoint %34 : !llvm.ptr to i64
    %68 = llvm.getelementptr %59[1] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %67, %68 : i64, !llvm.ptr
    %69 = llvm.ptrtoint %43 : !llvm.ptr to i64
    %70 = llvm.getelementptr %59[2] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %69, %70 : i64, !llvm.ptr
    %71 = llvm.ptrtoint %48 : !llvm.ptr to i64
    %72 = llvm.getelementptr %59[3] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %71, %72 : i64, !llvm.ptr
    %73 = llvm.ptrtoint %53 : !llvm.ptr to i64
    %74 = llvm.getelementptr %59[4] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %73, %74 : i64, !llvm.ptr
    llvm.store %3, %62 : i1, !llvm.ptr
    %75 = llvm.getelementptr %62[1] : (!llvm.ptr) -> !llvm.ptr, i1
    llvm.store %3, %75 : i1, !llvm.ptr
    %76 = llvm.getelementptr %62[2] : (!llvm.ptr) -> !llvm.ptr, i1
    llvm.store %3, %76 : i1, !llvm.ptr
    %77 = llvm.getelementptr %62[3] : (!llvm.ptr) -> !llvm.ptr, i1
    llvm.store %3, %77 : i1, !llvm.ptr
    %78 = llvm.getelementptr %62[4] : (!llvm.ptr) -> !llvm.ptr, i1
    llvm.store %3, %78 : i1, !llvm.ptr
    %79 = llvm.call @malloc(%61) : (i64) -> !llvm.ptr
    %80 = llvm.getelementptr %0[1] : (!llvm.ptr) -> !llvm.ptr, i1
    %81 = llvm.ptrtoint %80 : !llvm.ptr to i64
    %82 = llvm.call @malloc(%81) : (i64) -> !llvm.ptr
    llvm.call @dealloc_helper(%59, %59, %9, %6, %10, %65, %65, %9, %9, %10, %62, %62, %9, %6, %10, %79, %79, %9, %6, %10, %82, %82, %9, %9, %10) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    %83 = llvm.load %79 : !llvm.ptr -> i1
    llvm.cond_br %83, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    llvm.call @free(%18) : (!llvm.ptr) -> ()
    llvm.br ^bb8
  ^bb8:  // 2 preds: ^bb6, ^bb7
    %84 = llvm.getelementptr %79[1] : (!llvm.ptr) -> !llvm.ptr, i1
    %85 = llvm.load %84 : !llvm.ptr -> i1
    llvm.cond_br %85, ^bb9, ^bb10
  ^bb9:  // pred: ^bb8
    llvm.call @free(%29) : (!llvm.ptr) -> ()
    llvm.br ^bb10
  ^bb10:  // 2 preds: ^bb8, ^bb9
    %86 = llvm.getelementptr %79[2] : (!llvm.ptr) -> !llvm.ptr, i1
    %87 = llvm.load %86 : !llvm.ptr -> i1
    llvm.cond_br %87, ^bb11, ^bb12
  ^bb11:  // pred: ^bb10
    llvm.call @free(%42) : (!llvm.ptr) -> ()
    llvm.br ^bb12
  ^bb12:  // 2 preds: ^bb10, ^bb11
    %88 = llvm.getelementptr %79[3] : (!llvm.ptr) -> !llvm.ptr, i1
    %89 = llvm.load %88 : !llvm.ptr -> i1
    llvm.cond_br %89, ^bb13, ^bb14
  ^bb13:  // pred: ^bb12
    llvm.call @free(%47) : (!llvm.ptr) -> ()
    llvm.br ^bb14
  ^bb14:  // 2 preds: ^bb12, ^bb13
    %90 = llvm.getelementptr %79[4] : (!llvm.ptr) -> !llvm.ptr, i1
    %91 = llvm.load %90 : !llvm.ptr -> i1
    llvm.cond_br %91, ^bb15, ^bb16
  ^bb15:  // pred: ^bb14
    llvm.call @free(%52) : (!llvm.ptr) -> ()
    llvm.br ^bb16
  ^bb16:  // 2 preds: ^bb14, ^bb15
    llvm.call @free(%59) : (!llvm.ptr) -> ()
    llvm.call @free(%65) : (!llvm.ptr) -> ()
    llvm.call @free(%62) : (!llvm.ptr) -> ()
    llvm.call @free(%79) : (!llvm.ptr) -> ()
    llvm.call @free(%82) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @dealloc_helper(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: !llvm.ptr, %arg11: !llvm.ptr, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: !llvm.ptr, %arg16: !llvm.ptr, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: !llvm.ptr, %arg21: !llvm.ptr, %arg22: i64, %arg23: i64, %arg24: i64) attributes {sym_visibility = "private"} {
    %0 = llvm.mlir.constant(false) : i1
    %1 = llvm.mlir.constant(true) : i1
    %2 = llvm.mlir.constant(0 : index) : i64
    %3 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%2 : i64)
  ^bb1(%4: i64):  // 2 preds: ^bb0, ^bb2
    %5 = llvm.icmp "slt" %4, %arg8 : i64
    llvm.cond_br %5, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %6 = llvm.getelementptr %arg21[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i1
    llvm.store %0, %6 : i1, !llvm.ptr
    %7 = llvm.add %4, %3 : i64
    llvm.br ^bb1(%7 : i64)
  ^bb3:  // pred: ^bb1
    llvm.br ^bb4(%2 : i64)
  ^bb4(%8: i64):  // 2 preds: ^bb3, ^bb13
    %9 = llvm.icmp "slt" %8, %arg3 : i64
    llvm.cond_br %9, ^bb5, ^bb14
  ^bb5:  // pred: ^bb4
    %10 = llvm.getelementptr %arg1[%8] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %11 = llvm.load %10 : !llvm.ptr -> i64
    %12 = llvm.getelementptr %arg11[%8] : (!llvm.ptr, i64) -> !llvm.ptr, i1
    %13 = llvm.load %12 : !llvm.ptr -> i1
    llvm.br ^bb6(%2, %1 : i64, i1)
  ^bb6(%14: i64, %15: i1):  // 2 preds: ^bb5, ^bb9
    %16 = llvm.icmp "slt" %14, %arg8 : i64
    llvm.cond_br %16, ^bb7, ^bb10
  ^bb7:  // pred: ^bb6
    %17 = llvm.getelementptr %arg6[%14] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %18 = llvm.load %17 : !llvm.ptr -> i64
    %19 = llvm.icmp "eq" %18, %11 : i64
    llvm.cond_br %19, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %20 = llvm.getelementptr %arg21[%14] : (!llvm.ptr, i64) -> !llvm.ptr, i1
    %21 = llvm.load %20 : !llvm.ptr -> i1
    %22 = llvm.or %21, %13  : i1
    llvm.store %22, %20 : i1, !llvm.ptr
    llvm.br ^bb9
  ^bb9:  // 2 preds: ^bb7, ^bb8
    %23 = llvm.icmp "ne" %18, %11 : i64
    %24 = llvm.and %15, %23  : i1
    %25 = llvm.add %14, %3 : i64
    llvm.br ^bb6(%25, %24 : i64, i1)
  ^bb10:  // pred: ^bb6
    llvm.br ^bb11(%2, %15 : i64, i1)
  ^bb11(%26: i64, %27: i1):  // 2 preds: ^bb10, ^bb12
    %28 = llvm.icmp "slt" %26, %8 : i64
    llvm.cond_br %28, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    %29 = llvm.getelementptr %arg1[%26] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %30 = llvm.load %29 : !llvm.ptr -> i64
    %31 = llvm.icmp "ne" %30, %11 : i64
    %32 = llvm.and %27, %31  : i1
    %33 = llvm.add %26, %3 : i64
    llvm.br ^bb11(%33, %32 : i64, i1)
  ^bb13:  // pred: ^bb11
    %34 = llvm.and %27, %13  : i1
    %35 = llvm.getelementptr %arg16[%8] : (!llvm.ptr, i64) -> !llvm.ptr, i1
    llvm.store %34, %35 : i1, !llvm.ptr
    %36 = llvm.add %8, %3 : i64
    llvm.br ^bb4(%36 : i64)
  ^bb14:  // pred: ^bb4
    llvm.return
  }
}
