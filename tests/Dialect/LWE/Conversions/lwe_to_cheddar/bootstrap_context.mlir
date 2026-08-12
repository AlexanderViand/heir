// RUN: heir-opt --lwe-to-cheddar %s | FileCheck %s

#enc = #lwe.inverse_canonical_encoding<scaling_factor = 1099511627776>
#key = #lwe.key<>
#chain = #lwe.modulus_chain<elements = <36028797019488257 : i64, 1099512938497 : i64>, current = 1>
#ring_f64 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**65536>>
!rns = !rns.rns<!mod_arith.int<36028797019488257 : i64>, !mod_arith.int<1099512938497 : i64>>
#ring_rns = #polynomial.ring<coefficientType = !rns, polynomialModulus = <1 + x**65536>>
#ct_space = #lwe.ciphertext_space<ring = #ring_rns, encryption_type = mix>
!ct = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64, encoding = #enc>, ciphertext_space = #ct_space, key = #key, modulus_chain = #chain>

module attributes {
  scheme.ckks,
  ckks.schemeParam = #ckks.scheme_param<
    logN = 16,
    Q = [36028797019488257, 1099512938497],
    P = [2305843009211596801],
    logDefaultScale = 40
  >
} {
  // A non-boot helper receives the ordinary Context.
  // CHECK: func.func private @helper(
  // CHECK-SAME: !context
  // CHECK-NOT: !boot_context
  func.func private @helper(%ct: !ct) -> !ct {
    %sum = lwe.radd %ct, %ct : (!ct, !ct) -> !ct
    return %sum : !ct
  }

  // A transitive bootstrap caller carries both handles: BootContext for its
  // bootstrap operation and the base Context forwarded to non-boot callees.
  // CHECK: func.func @main(
  // CHECK-SAME: !context
  // CHECK-SAME: !boot_context
  // CHECK: cheddar.boot %{{.*}} :
  // CHECK: call @helper(%{{[^,]+}},
  func.func @main(%ct: !ct) -> !ct {
    %booted = ckks.bootstrap %ct : !ct -> !ct
    %result = func.call @helper(%booted) : (!ct) -> !ct
    return %result : !ct
  }
}
