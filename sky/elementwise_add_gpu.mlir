module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @elementwise_add(%lhs: memref<1024xf32>, %rhs: memref<1024xf32>, %out: memref<1024xf32>) kernel {
      %bx = gpu.block_id x
      %tx = gpu.thread_id x
      %bd = gpu.block_dim x
      %c1024 = arith.constant 1024 : index
      %base = arith.muli %bx, %bd : index
      %i = arith.addi %base, %tx : index
      %in_bounds = arith.cmpi ult, %i, %c1024 : index
      scf.if %in_bounds {
        %a = memref.load %lhs[%i] : memref<1024xf32>
        %b = memref.load %rhs[%i] : memref<1024xf32>
        %c = arith.addf %a, %b : f32
        memref.store %c, %out[%i] : memref<1024xf32>
      }
      gpu.return
    }
  }
}
