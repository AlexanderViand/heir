// 8x4 matrix-vector multiply as a CHEDDAR-dialect looped kernel.
//
// Hand-written at the post-(future)-LWE-to-Cheddar shape so this test
// exercises the tensor-of-ciphertext / memref-of-ciphertext path without
// requiring the upstream lowering. Mirrors what a CKKS-to-Cheddar lowering
// from `linalg.matvec` on `tensor<8x4xf32>` * `tensor<4xf32>` would yield
// once it bufferizes its output.
//
// Inputs:
//   %x : !cheddar.ciphertext  -- the 4-element input vector packed into
//                                slots [0..3] of a single ciphertext.
//   %W : tensor<8x!cheddar.plaintext>
//                             -- 8 row weight plaintexts; W[i] encodes
//                                the 4 weights of output row i into slots
//                                [0..3] (other slots zero).
//
// Output:
//   tensor<8x!cheddar.ciphertext>
//        -- result[i] is the row's dot product W_i . x in slot 0 (other
//           slots contain partial-sum garbage; downstream consumers must
//           ignore them or mask).
//
// Per-row computation (4 slots):
//   prod = mult_plain(x, W[i])                  -> [x0*w0, x1*w1, x2*w2, x3*w3, 0, ...]
//   s1   = hrot_add(prod, prod, distance=2)     -> [.+., .+., ?, ?, ...]
//   s2   = hrot_add(s1,   s1,   distance=1)     -> [SUM,  ?,   ?, ?, ...]
//
// Requires the caller's UserInterface to have rotation keys prepared for
// distances 1 and 2 at the operating level.
//
// Pipeline goal: bufferize tensor<8x!cheddar.ciphertext> /
// tensor<8x!cheddar.plaintext> to memref form, lower scf.for through
// EmitC, then translate to C++ that operates on `std::array<Ct,8>` /
// `std::array<Pt,8>` (or equivalent) with no intermediate allocation in
// the loop body.

func.func @matvec_8x4(%ctx: !cheddar.context,
                       %ui: !cheddar.user_interface,
                       %x: !cheddar.ciphertext,
                       %W: tensor<8x!cheddar.plaintext>)
    -> tensor<8x!cheddar.ciphertext> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %empty = tensor.empty() : tensor<8x!cheddar.ciphertext>
  %result = scf.for %i = %c0 to %c8 step %c1
      iter_args(%acc = %empty) -> tensor<8x!cheddar.ciphertext> {
    %wi = tensor.extract %W[%i] : tensor<8x!cheddar.plaintext>
    %prod = cheddar.mult_plain %ctx, %x, %wi
        : (!cheddar.context, !cheddar.ciphertext, !cheddar.plaintext)
            -> !cheddar.ciphertext
    %s1 = cheddar.hrot_add %ctx, %prod, %prod {distance = 2 : i64}
        : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext)
            -> !cheddar.ciphertext
    %s2 = cheddar.hrot_add %ctx, %s1, %s1 {distance = 1 : i64}
        : (!cheddar.context, !cheddar.ciphertext, !cheddar.ciphertext)
            -> !cheddar.ciphertext
    %updated = tensor.insert %s2 into %acc[%i] : tensor<8x!cheddar.ciphertext>
    scf.yield %updated : tensor<8x!cheddar.ciphertext>
  }
  return %result : tensor<8x!cheddar.ciphertext>
}
