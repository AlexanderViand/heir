// CHEDDAR-dialect compute kernels. The test driver does context/UI setup
// and encode/encrypt/decrypt boundary work directly via the CHEDDAR C++
// API; this file is just the compute that goes through the emitter.

func.func @add_kernel(%ctx: !cheddar.context,
                      %a: !cheddar.ciphertext,
                      %b: !cheddar.ciphertext) -> !cheddar.ciphertext {
  %r = cheddar.add %ctx, %a, %b
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
  return %r : !cheddar.ciphertext
}

func.func @sub_kernel(%ctx: !cheddar.context,
                      %a: !cheddar.ciphertext,
                      %b: !cheddar.ciphertext) -> !cheddar.ciphertext {
  %r = cheddar.sub %ctx, %a, %b
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
  return %r : !cheddar.ciphertext
}

func.func @add_plain_kernel(%ctx: !cheddar.context,
                            %a: !cheddar.ciphertext,
                            %p: !cheddar.plaintext) -> !cheddar.ciphertext {
  %r = cheddar.add_plain %ctx, %a, %p
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.plaintext) -> !cheddar.ciphertext
  return %r : !cheddar.ciphertext
}
