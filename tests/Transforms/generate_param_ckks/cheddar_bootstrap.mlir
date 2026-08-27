// RUN: heir-opt --generate-param-ckks="first-mod-bits=55 scaling-mod-bits=45 min-slot-count=8" %s | FileCheck %s

// A Cheddar bootstrap extends the generated Q chain with its CtS, EvalMod,
// and StC levels and records the split consumed by context configuration.

// CHECK: module attributes {
// CHECK-SAME: backend.cheddar
// CHECK-SAME: cheddar.boot.num_cts = 4 : i64
// CHECK-SAME: cheddar.boot.num_stc = 2 : i64
// CHECK-SAME: ckks.schemeParam = #ckks.scheme_param<
// CHECK-SAME: encryptionTechnique = extended
// CHECK-SAME: scheme.requested_slot_count = 8 : i64
module attributes {backend.cheddar, scheme.ckks} {
  func.func @bootstrap(%input: f32) -> f32 {
    %result = mgmt.bootstrap %input : f32
    return %result : f32
  }
}
