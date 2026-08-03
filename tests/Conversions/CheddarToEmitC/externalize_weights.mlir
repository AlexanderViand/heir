// RUN: heir-opt "--cheddar-externalize-weights=threshold=4" %s | FileCheck %s

// The --cheddar-externalize-weights pass wires externalized weight globals to
// a generated __load_constants() loader and keeps heir-translate from inlining
// a huge array initializer element-by-element (a splat <512x65536xf32> would
// otherwise expand to hundreds of MB of `0.0e+00f,` literals and OOM the host
// compiler). The weight blobs themselves are written earlier by the upstream
// externalize-constants pass; --convert-to-emitc lowers the resulting
// preprocessing.load_resource ops to uninitialized globals tagged with
// cheddar.load_path.
//
//   - tagged (externalized) weight -> heir_load_f32 entry in __load_constants,
//                                     tag removed.
//   - large all-zero splat         -> initializer dropped, NO loader entry
//                                     (relies on C++ static zero-init).
//   - large non-zero splat         -> initializer dropped, filled in
//                                     __load_constants.
//   - small constant               -> left inline.

module {
  // Externalized weight (tagged by ConvertLoadResource): loader entry, tag
  // removed.
  // CHECK: emitc.global static @constant_abc123 : !emitc.array<2x4xf32>{{$}}
  emitc.global static @constant_abc123 : !emitc.array<2x4xf32> {cheddar.load_path = "data/constant_abc123.bin"}

  // Large all-zero splat: initializer dropped.
  // CHECK: emitc.global static @zeros_big : !emitc.array<2x4xf32>{{$}}
  emitc.global static @zeros_big : !emitc.array<2x4xf32> = dense<0.000000e+00>

  // Large non-zero splat: initializer dropped (filled below).
  // CHECK: emitc.global static @twos_big : !emitc.array<2x4xf32>{{$}}
  emitc.global static @twos_big : !emitc.array<2x4xf32> = dense<2.000000e+00>

  // Large non-splat dense: left alone (externalize-constants should have
  // handled it before bufferization; this pass no longer writes blobs).
  // CHECK: emitc.global static @weight_inline : !emitc.array<2x4xf32> = dense<
  emitc.global static @weight_inline : !emitc.array<2x4xf32> = dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]>

  // Small constant: kept inline.
  // CHECK: emitc.global static @small : !emitc.array<2xf32> = dense<[1.000000e+00, 2.000000e+00]>
  emitc.global static @small : !emitc.array<2xf32> = dense<[1.0, 2.0]>
}

// Loader: blob load for the tagged weight, a fill loop for the non-zero
// splat, and nothing at all for the zero splat.
// CHECK: func.func @__load_constants()
// CHECK-DAG: heir_load_f32(\22data/constant_abc123.bin\22, reinterpret_cast<float*>(constant_abc123), 8)
// CHECK-DAG: reinterpret_cast<float*>(twos_big)[__i] = static_cast<float>(2)
// CHECK-NOT: zeros_big
