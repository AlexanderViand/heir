// 8-element dot product as a single CHEDDAR-dialect kernel.
//
// Inputs are packed as [a0..a7, 0, 0, ...] across the slot vector.  After
// `cheddar.hmult` we have a ciphertext whose slot i contains a[i]*b[i] for
// i < 8 and 0 elsewhere.  Three rotation+add steps then collapse the partial
// products: rot-by-4 + add gives [s0..s3] in slots 0..3, rot-by-2 + add
// gives [t0..t1] in slots 0..1, and a final rot-by-1 + add puts the full
// dot product in slot 0.
//
// Requires the caller's UserInterface to have rotation keys prepared for
// distances 1, 2, and 4 before the kernel runs.

func.func @dot_product_kernel(%ctx: !cheddar.context,
                              %ui: !cheddar.user_interface,
                              %mult_key: !cheddar.eval_key,
                              %a: !cheddar.ciphertext,
                              %b: !cheddar.ciphertext) -> !cheddar.ciphertext {
  %prod = cheddar.hmult %ctx, %a, %b, %mult_key {rescale = true}
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext, !cheddar.eval_key) -> !cheddar.ciphertext
  %s1 = cheddar.hrot_add %ctx, %prod, %prod {distance = 4 : i64}
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
  %s2 = cheddar.hrot_add %ctx, %s1, %s1 {distance = 2 : i64}
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
  %s3 = cheddar.hrot_add %ctx, %s2, %s2 {distance = 1 : i64}
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
  return %s3 : !cheddar.ciphertext
}
