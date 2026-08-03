// RUN: heir-opt --activation-canonicalizations %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// A torch ReLU imports as a linalg.generic carrying the polynomial-approximation
// domain on the generic op, with cmpf+select in its body. The float select
// must canonicalize to arith.maximumf AND inherit the generic's
// domain_lower/domain_upper so PolynomialApproximation reads the right domain.

// The domain must move onto the maximumf and NOT remain on the generic
// (leaving it on both collides during later activation lifting).
// CHECK: func.func @relu_generic
// CHECK: linalg.generic
// CHECK-NOT: domain_lower
// CHECK: %[[MAX:.*]] = arith.maximumf
<<<<<<< HEAD
// CHECK-SAME: domain_lower = -0.78033679723739624
// CHECK-SAME: domain_upper = 0.27823492884635925
=======
// CHECK-SAME: domain_lower = -7.803{{.*}}
// CHECK-SAME: domain_upper = 2.782{{.*}}
>>>>>>> 0fa9ea03f (ActivationCanonicalizations: forward ReLU domain attrs to maximumf)
// CHECK: linalg.yield %[[MAX]]
// CHECK: return
func.func @relu_generic(%arg0: tensor<1x3xf32>) -> tensor<1x3xf32> {
  %0 = tensor.empty() : tensor<1x3xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"], domain_lower = -0.78033679723739624 : f64, domain_upper = 0.27823492884635925 : f64} ins(%arg0 : tensor<1x3xf32>) outs(%0 : tensor<1x3xf32>) {
  ^bb0(%in: f32, %out: f32):
    %2 = arith.cmpf ugt, %in, %cst : f32
    %3 = arith.select %2, %in, %cst : f32
    linalg.yield %3 : f32
  } -> tensor<1x3xf32>
  return %1 : tensor<1x3xf32>
}
