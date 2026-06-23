// End-to-end test for the looped 8x4 matrix-vector multiply kernel emitted
// from matvec_8x4.mlir. Exercises tensor-of-ciphertext / tensor-of-plaintext
// patterns through scf.for, with a per-row `mult_plain + hrot_add + hrot_add`
// chain that reduces a 4-element packed dot product into slot 0.
//
// The exact emitted C++ signature depends on how bufferization + the
// cheddar-to-emitc emitter render `tensor<Nx!cheddar.*>` parameters. The
// expectation encoded here is `std::array<T,N>` for static-sized tensors of
// move-only payload types, plus by-out-reference for the result tensor.
// If the emitter ends up picking a different shape (e.g. raw pointers or
// `std::vector`), this signature needs to be updated to match.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "tests/Examples/cheddar/cheddar_test_fixture.h"

using cheddar::Context;
using cheddar::UserInterface;
using cheddar_test::Ct;
using cheddar_test::Pt;
using cheddar_test::word;

// Generated kernel: see matvec_8x4.mlir for the IR-level contract.
void matvec_8x4(Context<word>* ctx, UserInterface<word>* ui, const Ct& x,
                const std::array<Pt, 8>& W, std::array<Ct, 8>& result);

TEST(CheddarMatvec8x4E2E, RowwiseDotProducts) {
  auto f = cheddar_test::MakeFixture();
  // The kernel does hrot_add at distances 2 and 1 at the operating level.
  // mult_plain does NOT change the level, so we operate at top_level.
  f.ui->PrepareRotationKey(1, f.top_level);
  f.ui->PrepareRotationKey(2, f.top_level);

  // 4-element input vector, encrypted into a single ciphertext.
  std::vector<double> x_vec = {0.5, 0.25, 0.125, 0.0625};
  Ct x = cheddar_test::EncryptVec(f, x_vec, f.top_level);

  // 8x4 weight matrix as 8 plaintext rows. Picked so each row's expected
  // dot product is easy to eyeball.
  std::vector<std::vector<double>> W_vec = {
      {1.0, 0.0, 0.0, 0.0},    // row 0:  W_0 . x = 0.5
      {0.0, 1.0, 0.0, 0.0},    // row 1:  W_1 . x = 0.25
      {0.0, 0.0, 1.0, 0.0},    // row 2:  W_2 . x = 0.125
      {0.0, 0.0, 0.0, 1.0},    // row 3:  W_3 . x = 0.0625
      {1.0, 1.0, 1.0, 1.0},    // row 4:  W_4 . x = 0.9375
      {1.0, 0.5, 0.25, 0.0},   // row 5:  W_5 . x = 0.65625
      {0.5, 0.5, 0.5, 0.5},    // row 6:  W_6 . x = 0.46875
      {-1.0, 1.0, -1.0, 1.0},  // row 7:  W_7 . x = -0.3125
  };
  std::array<Pt, 8> W;
  for (size_t i = 0; i < W.size(); ++i) {
    W[i] = cheddar_test::EncodeVec(f, W_vec[i], f.top_level);
  }

  std::array<Ct, 8> result;
  matvec_8x4(f.ctx.get(), f.ui.get(), x, W, result);

  for (size_t i = 0; i < result.size(); ++i) {
    double expected = 0.0;
    for (size_t j = 0; j < x_vec.size(); ++j) {
      expected += W_vec[i][j] * x_vec[j];
    }
    auto slots = cheddar_test::DecryptVec(f, result[i], 1);
    EXPECT_NEAR(slots[0], expected, 0.01)
        << "Row " << i << ": expected " << expected << ", got " << slots[0];
  }
}
