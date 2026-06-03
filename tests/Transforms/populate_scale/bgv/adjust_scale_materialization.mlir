// RUN: heir-opt %s --secret-insert-mgmt-bgv=before-mul-include-first-mul --populate-scale-bgv | FileCheck %s

// Regression test for BGV adjust_scale materialization. The adjust_scale lowers
// to a multiplication by an all-ones plaintext (mgmt.init); the resulting
// ciphertext scale must be reduced modulo the plaintext modulus, because BGV
// scales live in Z_t (t = 65537 here). A prior bug annotated the mul with the
// raw, unreduced integer product inputScale * deltaScale, which then leaked
// into the BGV encoding scaling factor and corrupted decryption.
//
// Here the adjusted operand has scale 28458 and the all-ones plaintext has
// scale (delta) 10351, so the buggy code produced 28458 * 10351 = 294568758
// (>> t). The correct value is 294568758 mod 65537 = 45480.

module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 13, Q = [134250497, 17179967489, 35184372121601, 35184372744193, 36028797019389953], P = [36028797019488257], plaintextModulus = 65537>, scheme.bgv} {
  // CHECK: func @deep
  func.func @deep(%arg0: !secret.secret<i16>) -> !secret.secret<i16> {
    // The all-ones plaintext carries the (reduced) delta scale.
    // CHECK: %[[ONES:.*]] = mgmt.init
    // CHECK-SAME: scale = 10351
    %0 = secret.generic(%arg0 : !secret.secret<i16>) {
    ^body(%input0: i16):
      %1 = arith.muli %input0, %input0 : i16
      %2 = arith.muli %1, %1 : i16
      // Adding the shallow product %1 to the deep product %2 forces a scale
      // adjustment of %1, materialized as a multiply by the all-ones plaintext.
      // CHECK: arith.muli %{{[^,]*}}, %[[ONES]]
      // CHECK-SAME: scale = 45480
      %3 = arith.addi %2, %1 : i16
      secret.yield %3 : i16
    } -> !secret.secret<i16>
    return %0 : !secret.secret<i16>
  }
}
