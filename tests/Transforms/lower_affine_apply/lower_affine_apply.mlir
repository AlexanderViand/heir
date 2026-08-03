// RUN: heir-opt %s --lower-affine-apply --split-input-file | FileCheck %s

// affine.apply is expanded to scalar arith, while affine.for is left intact
// (the scheme backend emitters print affine.for directly but have no printer
// for affine.apply).

#map = affine_map<(d0) -> (d0 * 3 + 1)>

// CHECK: func.func @apply_in_loop
// CHECK:         affine.for %[[I:.*]] = 0 to 2 {
// CHECK-NOT:       affine.apply
// CHECK:           %[[MUL:.*]] = arith.muli %[[I]], %{{.*}} : index
// CHECK:           %[[IDX:.*]] = arith.addi %[[MUL]], %{{.*}} : index
// CHECK:           memref.store %{{.*}}, %{{.*}}[%[[IDX]]]
// CHECK:         }
func.func @apply_in_loop(%m: memref<8xf32>, %c: f32) {
  affine.for %i = 0 to 2 {
    %0 = affine.apply #map(%i)
    memref.store %c, %m[%0] : memref<8xf32>
  }
  return
}

// -----

// A standalone affine.apply with both a dim and a symbol operand.

#map2 = affine_map<(d0)[s0] -> (d0 + s0 * 2)>

// CHECK: func.func @apply_with_symbol
// CHECK-NOT:     affine.apply
// CHECK:         arith.muli
// CHECK:         arith.addi
// CHECK:         return %{{.*}} : index
func.func @apply_with_symbol(%d: index, %s: index) -> index {
  %0 = affine.apply #map2(%d)[%s]
  return %0 : index
}
