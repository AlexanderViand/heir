// RUN: heir-opt --split-input-file --generate-param-ckks="first-mod-bits=50 validate-first-mod-bits=false scaling-mod-bits=45" %s | FileCheck %s

// CHECK: module attributes
// CHECK-SAME: Q = [{{[0-9]+}}]
module {
  func.func @no_extra_cost(%arg0: !secret.secret<f16> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) -> (!secret.secret<f16> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) {
    %0 = secret.generic(%arg0: !secret.secret<f16> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) attrs = {arg0 = {mgmt.mgmt = #mgmt.mgmt<level = 0>}} {
    ^body(%input0: f16):
      %1 = arith.addf %input0, %input0 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : f16
      secret.yield %1 : f16
    } -> (!secret.secret<f16> {mgmt.mgmt = #mgmt.mgmt<level = 0>})
    return %0 : !secret.secret<f16>
  }
}

// -----

// CHECK: module attributes
// CHECK-SAME: Q = [{{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}]
module {
  func.func @extra_cost(%arg0: !secret.secret<f16> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) -> (!secret.secret<f16> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) {
    %0 = secret.generic(%arg0: !secret.secret<f16> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) attrs = {arg0 = {mgmt.mgmt = #mgmt.mgmt<level = 0>}, orion.level_cost_ub = 2 : i64} {
    ^body(%input0: f16):
      %1 = arith.addf %input0, %input0 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : f16
      secret.yield %1 : f16
    } -> (!secret.secret<f16> {mgmt.mgmt = #mgmt.mgmt<level = 0>})
    return %0 : !secret.secret<f16>
  }
}
