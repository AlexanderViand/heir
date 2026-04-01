// RUN: heir-opt --annotate-mgmt=auto-relinearize=true %s | FileCheck %s

// With auto-relinearize, secret-secret mul should produce dimension 2 (not 3).
// Dimension 2 is the default and is omitted from the MgmtAttr printing.

// CHECK: func @auto_relin_dimension
func.func @auto_relin_dimension(
    %arg0: !secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 2>},
    %arg1: !secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 2>}
  ) -> (!secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) {
  %1 = secret.generic(
      %arg0: !secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 2>},
      %arg1: !secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 2>}
    ) {
    ^body(%input0: tensor<1024xi16>, %input1: tensor<1024xi16>):
      // With auto-relin: dimension is 2 (default, omitted), NOT 3
      // CHECK: arith.muli
      // CHECK-SAME: {mgmt.mgmt = #mgmt.mgmt<level = 0>}
      // CHECK-NOT: dimension = 3
      %2 = arith.muli %input0, %input1 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<1024xi16>
      secret.yield %2 : tensor<1024xi16>
  } -> (!secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 0>})
  return %1 : !secret.secret<tensor<1024xi16>>
}
