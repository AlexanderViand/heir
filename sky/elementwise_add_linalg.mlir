#map = affine_map<(d0) -> (d0)>

module {
  func.func @elementwise_add(%lhs: memref<16384xi64>, %rhs: memref<16384xi64>, %out: memref<16384xi64>) {
    linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel"]
    } ins(%lhs, %rhs : memref<16384xi64>, memref<16384xi64>) outs(%out : memref<16384xi64>) {
    ^bb0(%a: i64, %b: i64, %c: i64):
      %sum = arith.addi %a, %b : i64
      linalg.yield %sum : i64
    }
    return
  }
}
