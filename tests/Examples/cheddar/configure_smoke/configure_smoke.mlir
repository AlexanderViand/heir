!ciphertext = !cheddar.ciphertext
!context = !cheddar.context
!evk_map = !cheddar.evk_map

module attributes {
  ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797018652673, 1125899907366913, 1125899907760129], P = [1152921504606994433], logDefaultScale = 45>,
  scheme.actual_slot_count = 4096 : i64
} {
  func.func @rotate(%ctx: !context, %evk: !evk_map, %ct: tensor<!ciphertext>) -> tensor<!ciphertext> {
    %d0 = tensor.empty() : tensor<!ciphertext>
    %rot = cheddar.hrot %ctx, %evk, %ct, %d0 {level = 2 : i64, static_distance = 7 : i64} : (!context, !evk_map, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %d1 = tensor.empty() : tensor<!ciphertext>
    %result = cheddar.hrot_add %ctx, %evk, %rot, %ct, %d1 {distance = 2 : i64, level = 2 : i64} : (!context, !evk_map, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    return %result : tensor<!ciphertext>
  }
}
