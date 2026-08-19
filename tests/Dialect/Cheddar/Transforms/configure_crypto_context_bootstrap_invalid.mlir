// RUN: not heir-opt --cheddar-configure-crypto-context=entry-function=main %s 2>&1 | FileCheck %s

!boot_context = !cheddar.boot_context
!ciphertext = !cheddar.ciphertext
!evk_map = !cheddar.evk_map

module attributes {
  ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797018652673, 1125899907366913, 1125899907760129], P = [1152921504606994433], logDefaultScale = 45>,
  scheme.actual_slot_count = 4096 : i64,
  scheme.requested_slot_count = 4 : i64
} {
  // CHECK: error: 'func.func' op CHEDDAR bootstrap modulus chain is too short: 3 Q primes cannot cover 3 CtS + 8 EvalMod + 3 StC levels
  func.func @main(%ctx: !boot_context, %ct: tensor<!ciphertext>, %evk: !evk_map) -> tensor<!ciphertext> {
    %dest = bufferization.alloc_tensor() : tensor<!ciphertext>
    %result = cheddar.boot %ctx, %ct, %evk, %dest : (!boot_context, tensor<!ciphertext>, !evk_map, tensor<!ciphertext>) -> tensor<!ciphertext>
    return %result : tensor<!ciphertext>
  }
}
