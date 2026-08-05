#ifndef LIB_TARGET_COMPILATIONTARGET_HEIRTARGET_TD_
#define LIB_TARGET_COMPILATIONTARGET_HEIRTARGET_TD_

class CompilationTarget<string name> {
  string backendName = name;
  int bootstrapLevelsConsumed = 0;
  int has_kernel_chebyshev = 0;
  // Whether the backend can represent mgmt.adjust_scale (a same-level scalar
  // scale correction). Backends with one fixed canonical scale per level
  // (cheddar) cannot; secret-insert-mgmt then resolves cross-level mismatches
  // with level_reduce only and rejects any adjust_scale.
  int can_emit_adjust_scale = 1;
}

#endif // LIB_TARGET_COMPILATIONTARGET_HEIRTARGET_TD_
