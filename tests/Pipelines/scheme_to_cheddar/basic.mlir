// RUN: heir-opt --scheme-to-cheddar="entry-function=main" %s | FileCheck %s

#encoding = #lwe.inverse_canonical_encoding<scaling_factor = 1099511627776>
#key = #lwe.key<>
#chain = #lwe.modulus_chain<elements = <36028797019488257 : i64, 1099512938497 : i64>, current = 1>
#ring_f64 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**65536>>
!rns = !rns.rns<!mod_arith.int<36028797019488257 : i64>, !mod_arith.int<1099512938497 : i64>>
#ring_rns = #polynomial.ring<coefficientType = !rns, polynomialModulus = <1 + x**65536>>
#ct_space = #lwe.ciphertext_space<ring = #ring_rns, encryption_type = mix>
!ct = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64, encoding = #encoding>, ciphertext_space = #ct_space, key = #key, modulus_chain = #chain>

// CHECK: module attributes {
// CHECK-SAME: backend.cheddar
// CHECK-SAME: cheddar.P = array<i64: 2305843009211596801>
// CHECK-SAME: cheddar.Q = array<i64: 36028797019488257, 1099512938497>
// CHECK-NOT: scheme.ckks
module attributes {
  backend.cheddar,
  scheme.ckks,
  ckks.schemeParam = #ckks.scheme_param<
    logN = 16,
    Q = [36028797019488257, 1099512938497],
    P = [2305843009211596801],
    logDefaultScale = 40
  >
} {
  // CHECK: func.func @main(
  // CHECK: cheddar.add
  // CHECK-NOT: lwe.
  // CHECK-NOT: ckks.
  func.func @main(%lhs: !ct, %rhs: !ct) -> !ct {
    %result = lwe.radd %lhs, %rhs : (!ct, !ct) -> !ct
    return %result : !ct
  }

  // CHECK: func.func @main__setup
  // CHECK: cheddar.make_parameter
  // CHECK: cheddar.create_context
  // CHECK: func.func @main__keygen
  // CHECK: cheddar.create_user_interface
  // CHECK: func.func @main__configure
  // CHECK: call @main__setup
  // CHECK: call @main__keygen
}
