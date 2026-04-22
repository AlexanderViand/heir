// End-to-end matvec test using CHEDDAR GPU FHE backend.
// Computes y = Ax + b where A (16x16) and b (16) are constant-embedded
// in the MLIR and x (16) is the encrypted input.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "tests/Examples/cheddar/ckks/matvec/matvec_lib.h"

using namespace cheddar;
using word = uint64_t;
using Complex = std::complex<double>;

TEST(CheddarMatvec, EndToEnd) {
  std::cout << "=== CHEDDAR Matvec 16x16 Test ===" << std::endl;

  std::cout << "[1/6] Creating context..." << std::flush;
  auto [context, ui] = __configure();
  std::cout << " done" << std::endl;

  std::cout << "[2/6] Keys ready..." << std::endl;

  // Input vector: x = linspace(0, 1, 16)
  std::vector<double> input(16);
  for (int i = 0; i < 16; ++i) {
    input[i] = static_cast<double>(i) / 15.0;
  }

  std::cout << "[3/6] Encrypting input..." << std::flush;
  auto& encoder = context->encoder_;
  auto ct_input = matvec__encrypt__arg0(context, encoder, ui, input, ui);
  std::cout << " done" << std::endl;

  std::cout << "[4/6] Computing matvec on GPU..." << std::flush;
  auto result_ct = matvec(context, encoder, ui, ct_input);
  std::cout << " done" << std::endl;

  std::cout << "[5/6] Decrypting..." << std::flush;
  auto result = matvec__decrypt__result0(context, encoder, ui, result_ct, ui);
  std::cout << " done" << std::endl;

  // Expected output y = Ax + b, computed in cleartext from the constant
  // matrix and bias embedded in matvec.mlir.
  std::vector<double> expected = {
      -2.3821208573, 0.6871068985,  -1.2167895839, -0.4125589382,
      2.3051484933,  -1.4409237797, -1.4009754067, 2.8419418401,
      1.3630157702,  -2.0137318864, 0.5629377979,  -0.9730263679,
      1.0751746167,  0.6354784853,  -0.2588269626, -0.4819815874};

  std::cout << "[6/6] Checking result..." << std::endl;
  ASSERT_EQ(result.size(), 16u) << "Expected 16-element output";

  for (int i = 0; i < 16; ++i) {
    std::cout << "  y[" << i << "]  expected=" << expected[i]
              << "  got=" << result[i]
              << "  error=" << std::abs(result[i] - expected[i]) << std::endl;
    EXPECT_TRUE(std::isfinite(result[i])) << "y[" << i << "] is not finite";
    EXPECT_NEAR(result[i], expected[i], 0.5)
        << "y[" << i << "] deviates too much from expected";
  }

  std::cout << "=== CHEDDAR Matvec Test Complete ===" << std::endl;
}
