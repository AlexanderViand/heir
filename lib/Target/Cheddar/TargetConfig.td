include "lib/Target/CompilationTarget/HEIRTarget.td"

def Cheddar : CompilationTarget<"cheddar"> {
  // In HEIR's level model a cheddar bootstrap consumes no compute levels: the
  // bootstrap-circuit levels (CtS/EvalMod/StC) are added on top of the compute
  // chain by generate-param-ckks (kCheddarBootOverhead), and the bootstrap
  // output lands back at the compute max level.
  let bootstrapLevelsConsumed = 0;
}
