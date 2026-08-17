func.func @load_float_resource() -> f32 {
  %resource = memref.alloc() : memref<2x2xf32>
  preprocessing.load_resource "weights.bin" into %resource
      : (memref<2x2xf32>) -> ()
  %c0 = arith.constant 0 : index
  %value = memref.load %resource[%c0, %c0] : memref<2x2xf32>
  return %value : f32
}

func.func @load_integer_resource() -> i64 {
  %resource = memref.alloc() : memref<4xi64>
  preprocessing.load_resource "indices.bin" into %resource
      : (memref<4xi64>) -> ()
  %c0 = arith.constant 0 : index
  %value = memref.load %resource[%c0] : memref<4xi64>
  return %value : i64
}
