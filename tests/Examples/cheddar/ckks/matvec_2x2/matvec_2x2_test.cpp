// End-to-end 2x2 matvec test using CHEDDAR GPU FHE backend.
// Matrix [[1,2],[3,4]] * [1,2] = [5, 11]

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "tests/Examples/cheddar/ckks/matvec_2x2/matvec_2x2_lib.h"

using namespace cheddar;
using word = uint64_t;
using Complex = std::complex<double>;

TEST(CheddarMatvec2x2, EndToEnd) {
  std::cout << "=== CHEDDAR Matvec 2x2 Test ===" << std::endl;

  std::cout << "[1/6] Creating context..." << std::flush;
  auto [context, ui] = __configure();
  std::cout << " done" << std::endl;

  std::cout << "[2/6] Keys ready..." << std::endl;

  // Input vector
  std::vector<double> input = {1.0, 2.0};

  // Expected output: [[1,2],[3,4]] * [1,2] = [1*1+2*2, 3*1+4*2] = [5, 11]
  std::vector<double> expected = {5.0, 11.0};

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

  std::cout << "[6/6] Checking result..." << std::endl;
  ASSERT_EQ(result.size(), 2u) << "Expected 2-element output";

  for (int i = 0; i < 2; ++i) {
    std::cout << "  y[" << i << "]  expected=" << expected[i]
              << "  got=" << result[i]
              << "  error=" << std::abs(result[i] - expected[i]) << std::endl;
    EXPECT_TRUE(std::isfinite(result[i])) << "y[" << i << "] is not finite";
    EXPECT_NEAR(result[i], expected[i], 0.1)
        << "y[" << i << "] deviates too much from expected";
  }

  std::cout << "=== CHEDDAR Matvec 2x2 Test Complete ===" << std::endl;
}
