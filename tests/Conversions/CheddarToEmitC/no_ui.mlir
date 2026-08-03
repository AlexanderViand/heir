// RUN: heir-opt --convert-to-emitc --verify-diagnostics %s

// HRot/HRotAdd/HConj/HConjAdd discover the UserInterface from the enclosing
// function's argument list at lowering time. A function that lacks such an
// argument cannot be legalized: the rotation pattern emits its own diagnostic
// and the op fails to legalize.

!ciphertext = !cheddar.ciphertext
!context = !cheddar.context

func.func @hrot_without_ui(%ctx: !context, %ct: memref<!ciphertext>,
                           %out: memref<!ciphertext>) {
  // expected-error @below {{'cheddar.hrot' op enclosing function is missing UserInterface arg}}
  // expected-error @below {{failed to legalize operation 'cheddar.hrot'}}
  cheddar.hrot %ctx, %ct, %out {static_distance = 1 : i64}
      : (!context, memref<!ciphertext>, memref<!ciphertext>) -> ()
  return
}
