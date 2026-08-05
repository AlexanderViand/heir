include "lib/Target/CompilationTarget/HEIRTarget.td"

def Cheddar : CompilationTarget<"cheddar"> {
  // In HEIR's level model a cheddar bootstrap consumes no compute levels: the
  // bootstrap-circuit levels (CtS/EvalMod/StC) are added on top of the compute
  // chain by generate-param-ckks (kCheddarBootOverhead), and the bootstrap
  // output lands back at the compute max level.
  let bootstrapLevelsConsumed = 0;
  // CHEDDAR has one fixed canonical scale per level (GetScale(level)), so a
  // same-level scalar scale correction is unrepresentable; secret-insert-mgmt
  // resolves cross-level mismatches with level_reduce (cheddar.level_down)
  // instead and rejects any adjust_scale.
  let can_emit_adjust_scale = 0;
}
