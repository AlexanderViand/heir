// RUN: not heir-opt "--generate-param-ckks=ring-dim=1024" %s 2>&1 | FileCheck %s --check-prefix=REJECT
// RUN: heir-opt "--generate-param-ckks=ring-dim=1024 allow-insecure-ring-dim=true" %s 2>&1 | FileCheck %s --check-prefix=ALLOW

// REJECT: forced ring dimension 1024 is below the 128-bit classic security minimum
// REJECT-SAME: explicitly set allow-insecure-ring-dim=true for benchmarking

// ALLOW: warning: INSECURE BENCHMARK PARAMETERS: forced ring dimension 1024
// ALLOW: module attributes {
// ALLOW-SAME: scheme.insecure_parameters

module {
  func.func @identity(%arg0: !secret.secret<f16> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) -> (!secret.secret<f16> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) {
    return %arg0 : !secret.secret<f16>
  }
}
