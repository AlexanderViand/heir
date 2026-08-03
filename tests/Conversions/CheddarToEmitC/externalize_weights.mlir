// RUN: rm -rf %t.d && heir-opt "--cheddar-externalize-weights=data-dir=%t.d threshold=4" %s | FileCheck %s

// The --cheddar-externalize-weights pass decides how each large weight global
// reaches the emitted C++ so heir-translate never inlines a huge array
// initializer element-by-element (a splat <512x65536xf32> would otherwise
// expand to hundreds of MB of `0.0e+00f,` literals and OOM the host compiler).
//
//   - large non-splat weight  -> initializer stripped, raw bytes written to
//                                 <data-dir>/<name>.bin, loaded at runtime.
//   - large all-zero splat    -> initializer dropped, NO loader entry (relies
//                                 on C++ static zero-initialization).
//   - large non-zero splat    -> initializer dropped, filled in __load_constants.
//   - small constant          -> left inline.

module {
  // Large non-splat: initializer stripped (no `= dense` -> end of line).
  // CHECK: emitc.global static @weight_big : !emitc.array<2x4xf32>{{$}}
  emitc.global static @weight_big : !emitc.array<2x4xf32> = dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]>

  // Large all-zero splat: initializer dropped.
  // CHECK: emitc.global static @zeros_big : !emitc.array<2x4xf32>{{$}}
  emitc.global static @zeros_big : !emitc.array<2x4xf32> = dense<0.000000e+00>

  // Large non-zero splat: initializer dropped (filled below).
  // CHECK: emitc.global static @twos_big : !emitc.array<2x4xf32>{{$}}
  emitc.global static @twos_big : !emitc.array<2x4xf32> = dense<2.000000e+00>

  // Small constant: kept inline.
  // CHECK: emitc.global static @small : !emitc.array<2xf32> = dense<[1.000000e+00, 2.000000e+00]>
  emitc.global static @small : !emitc.array<2xf32> = dense<[1.0, 2.0]>
}

// Loader: blob load for the non-splat weight, a fill loop for the non-zero
// splat, and nothing at all for the zero splat.
// CHECK: func.func @__load_constants()
// CHECK-DAG: heir_load_f32(\22data/weight_big.bin\22, reinterpret_cast<float*>(weight_big), 8)
// CHECK-DAG: reinterpret_cast<float*>(twos_big)[__i] = static_cast<float>(2)
// CHECK-NOT: zeros_big
