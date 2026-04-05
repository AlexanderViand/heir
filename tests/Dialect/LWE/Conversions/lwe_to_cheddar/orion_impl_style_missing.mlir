// RUN: not heir-opt --lwe-to-cheddar %s 2>&1 | FileCheck %s

// CHECK: error: 'orion.chebyshev' op requires Orion implementation style `opaque`, but no `orion.impl_style` annotation is present

!Z17_i64 = !mod_arith.int<17 : i64>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 16 : i64>
#key = #lwe.key<>
#modulus_chain_L0_C0 = #lwe.modulus_chain<elements = <17 : i64>, current = 0>
#ring_f64_1_x8 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**8>>
!rns_L0 = !rns.rns<!Z17_i64>
#ring_rns_L0_1_x8 = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**8>>
#ciphertext_space_L0 = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x8, encryption_type = mix>
!ct_L0 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x8, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L0, key = #key, modulus_chain = #modulus_chain_L0_C0>

module attributes {scheme.ckks, ckks.schemeParam = #ckks.scheme_param<logN = 3, Q = [17], P = [257], logDefaultScale = 4>} {
  func.func @chebyshev(%ct: !ct_L0) -> !ct_L0 {
    %ct_0 = orion.chebyshev %ct {coefficients = [0.0, 0.75, 0.0, 0.25], domain_end = 1.000000e+00 : f64, domain_start = -1.000000e+00 : f64} : (!ct_L0) -> !ct_L0
    return %ct_0 : !ct_L0
  }
}
