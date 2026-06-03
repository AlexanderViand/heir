// CHEDDAR-dialect kernels exercising ciphertext multiplication. The result of
// `cheddar.mult` is a degree-3 ciphertext at the same level as its inputs;
// `cheddar.relinearize_rescale` brings it back to degree-2 and drops one
// level, so callers see a normal ciphertext at level-1.

func.func @mult_kernel(%ctx: !cheddar.context, %mult_key: !cheddar.eval_key,
                       %a: !cheddar.ciphertext,
                       %b: !cheddar.ciphertext) -> !cheddar.ciphertext {
  %prod = cheddar.mult %ctx, %a, %b
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext) -> !cheddar.ciphertext
  %r = cheddar.relinearize_rescale %ctx, %prod, %mult_key
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.eval_key) -> !cheddar.ciphertext
  return %r : !cheddar.ciphertext
}

// Fused single-kernel variant: `cheddar.hmult` does mult + relin + rescale
// in one GPU launch.
func.func @hmult_kernel(%ctx: !cheddar.context, %mult_key: !cheddar.eval_key,
                        %a: !cheddar.ciphertext,
                        %b: !cheddar.ciphertext) -> !cheddar.ciphertext {
  %r = cheddar.hmult %ctx, %a, %b, %mult_key {rescale = true}
      : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext, !cheddar.eval_key) -> !cheddar.ciphertext
  return %r : !cheddar.ciphertext
}
