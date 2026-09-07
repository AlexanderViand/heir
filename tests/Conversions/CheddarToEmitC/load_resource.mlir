// RUN: heir-opt --ownership-based-buffer-deallocation --canonicalize --buffer-deallocation-simplification --bufferization-lower-deallocations --cheddar-emitc-boundary --convert-to-emitc=filter-dialects=cheddar,arith,scf %s | FileCheck %s

// CHECK: emitc.include "lib/Runtime/CleartextResource.h"
// CHECK: func.func @load_resource
// CHECK: emitc.call_opaque "malloc"
// CHECK: %[[DATA:.*]] = emitc.cast
// CHECK: emitc.call_opaque "heir::loadResource"(%[[DATA]])
// CHECK-SAME: weights.bin
// CHECK-SAME: #emitc.opaque<"4">
// CHECK-SAME: template_args = [f32]
// CHECK: %[[ELEMENT:.*]] = emitc.subscript %[[DATA]]
// CHECK: emitc.load %[[ELEMENT]]
// CHECK: call_opaque "free"(%[[DATA]])
func.func @load_resource() -> f32 {
  %resource = memref.alloc() : memref<2x2xf32>
  preprocessing.load_resource "weights.bin" into %resource
      : (memref<2x2xf32>) -> ()
  %c0 = arith.constant 0 : index
  %value = memref.load %resource[%c0, %c0] : memref<2x2xf32>
  return %value : f32
}

// Integer resources use the same caller-owned storage ABI and runtime helper.
// CHECK: func.func @load_integer_resource
// CHECK: emitc.call_opaque "heir::loadResource"(%{{[^)]*}})
// CHECK-SAME: indices.bin
// CHECK-SAME: template_args = [i64]
// CHECK: emitc.load
func.func @load_integer_resource() -> i64 {
  %resource = memref.alloc() : memref<4xi64>
  preprocessing.load_resource "indices.bin" into %resource
      : (memref<4xi64>) -> ()
  %c0 = arith.constant 0 : index
  %value = memref.load %resource[%c0] : memref<4xi64>
  return %value : i64
}

// Heap-backed resource subviews use flat pointer arithmetic.
// CHECK: func.func @load_resource_subview
// CHECK: %[[MATRIX:.*]] = emitc.cast
// CHECK: emitc.call_opaque "heir::loadResource"(%[[MATRIX]])
// CHECK-SAME: matrix.bin
// CHECK: %[[OFFSET:.*]] = emitc.literal "4" : !emitc.opaque<"std::ptrdiff_t">
// CHECK: %[[ROW:.*]] = emitc.add %[[MATRIX]], %[[OFFSET]]
// CHECK-NOT: unrealized_conversion_cast
// CHECK: %[[ELEMENT:.*]] = emitc.subscript %[[ROW]]
// CHECK: emitc.load %[[ELEMENT]]
func.func @load_resource_subview() -> f32 {
  %resource = memref.alloc() : memref<4x4xf32>
  preprocessing.load_resource "matrix.bin" into %resource
      : (memref<4x4xf32>) -> ()
  %row = memref.subview %resource[1, 0] [1, 4] [1, 1]
      : memref<4x4xf32> to memref<1x4xf32, strided<[4, 1], offset: 4>>
  %c0 = arith.constant 0 : index
  %value = memref.load %row[%c0, %c0]
      : memref<1x4xf32, strided<[4, 1], offset: 4>>
  return %value : f32
}

// The 4096-byte resource exceeds the configured threshold. It therefore stays
// a memref.alloc, lowers through the standard malloc path, and is freed after
// its last use by the standard ownership-based deallocation pipeline.
// CHECK: func.func @load_large_resource
// CHECK: emitc.call_opaque "malloc"
// CHECK: %[[HEAP:.*]] = emitc.cast
// CHECK: emitc.call_opaque "heir::loadResource"(%[[HEAP]])
// CHECK-SAME: large.bin
// CHECK-SAME: #emitc.opaque<"1024">
// CHECK: call_opaque "free"(%[[HEAP]])
func.func @load_large_resource() -> f32 {
  %resource = memref.alloc() : memref<1024xf32>
  preprocessing.load_resource "large.bin" into %resource
      : (memref<1024xf32>) -> ()
  %c0 = arith.constant 0 : index
  %value = memref.load %resource[%c0] : memref<1024xf32>
  return %value : f32
}
