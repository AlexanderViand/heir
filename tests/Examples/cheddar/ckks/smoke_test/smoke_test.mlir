// A simple CHEDDAR smoke test: add two ciphertexts and return the result.
// This is already in the cheddar dialect (post-lowering).
module {
  func.func @smoke_add(
      %ctx: !cheddar.context,
      %encoder: !cheddar.encoder,
      %ui: !cheddar.user_interface,
      %ct0: !cheddar.ciphertext,
      %ct1: !cheddar.ciphertext) -> !cheddar.ciphertext {
    %result = cheddar.add %ctx, %ct0, %ct1 : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
    return %result : !cheddar.ciphertext
  }
}
