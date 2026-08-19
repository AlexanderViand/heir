// RUN: heir-opt --cheddar-emitc-boundary --split-input-file --verify-diagnostics %s

// The getter-style setup ops return a const reference to a move-only /
// non-assignable CHEDDAR value (EvkMap, EvaluationKey, Encoder), which can't be
// materialised into a local without a copy. The lowering rejects them rather
// than emit uncompilable C++. Real kernels pass these as function arguments or
// look them up inline (like HRot's rotation-key lookup). (create_user_interface
// is now supported -- it is a DPS op producing an owning unique_ptr.)

func.func @get_evk_map(%ui: !cheddar.user_interface) -> !cheddar.evk_map {
  // expected-error @below {{lowering of 'cheddar.get_evk_map' is not supported}}
  %m = cheddar.get_evk_map %ui : (!cheddar.user_interface) -> !cheddar.evk_map
  return %m : !cheddar.evk_map
}

// -----

func.func @get_mult_key(%ui: !cheddar.user_interface) -> !cheddar.eval_key {
  // expected-error @below {{lowering of 'cheddar.get_mult_key' is not supported}}
  %k = cheddar.get_mult_key %ui : (!cheddar.user_interface) -> !cheddar.eval_key
  return %k : !cheddar.eval_key
}

// -----

func.func @get_encoder(%ctx: !cheddar.context) -> !cheddar.encoder {
  // expected-error @below {{lowering of 'cheddar.get_encoder' is not supported}}
  %e = cheddar.get_encoder %ctx : (!cheddar.context) -> !cheddar.encoder
  return %e : !cheddar.encoder
}

// -----

// The lowering uses one function-local static Parameter so the Context can
// safely retain its reference after the configure function returns.
// expected-error @below {{cheddar-to-emitc supports at most one cheddar.make_parameter per function}}
func.func @duplicate_parameter() {
  %p0 = cheddar.make_parameter {logN = 14 : i64, logScale = 45 : i64, mainPrimes = array<i64: 1, 2>, auxPrimes = array<i64: 3>} : !cheddar.parameter
  %p1 = cheddar.make_parameter {logN = 14 : i64, logScale = 45 : i64, mainPrimes = array<i64: 1, 2>, auxPrimes = array<i64: 3>} : !cheddar.parameter
  return
}
