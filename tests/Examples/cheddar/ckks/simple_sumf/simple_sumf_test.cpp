// End-to-end simple sum test using CHEDDAR GPU FHE backend.
// Computes the sum of a 32-element f32 tensor.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include "tests/Examples/cheddar/ckks/simple_sumf/simple_sumf_lib.h"

using namespace cheddar;
using word = uint64_t;
using Complex = std::complex<double>;

TEST(CheddarSimpleSumf, EndToEnd) {
  std::cout << "=== CHEDDAR Simple Sum (f32) Test ===" << std::endl;

  std::cout << "[1/6] Creating context..." << std::flush;
  auto [context, ui] = __configure();
  std::cout << " done" << std::endl;

  std::cout << "[2/6] Keys ready..." << std::endl;

  // Input: 32 values from 0.1 to 3.2 in steps of 0.1
  std::vector<double> arg0(32);
  for (int i = 0; i < 32; ++i) {
    arg0[i] = 0.1 * (i + 1);
  }

  // Expected: sum of 0.1 + 0.2 + ... + 3.2 = 0.1 * (1+2+...+32) = 0.1 * 528
  // = 52.8
  double expected = 0.0;
  for (int i = 0; i < 32; ++i) {
    expected += arg0[i];
  }

  std::cout << "[3/6] Encrypting input..." << std::flush;
  auto& encoder = context->encoder_;
  auto ct0 = simple_sum__encrypt__arg0(context, encoder, ui, arg0, ui);
  std::cout << " done" << std::endl;

  std::cout << "[4/6] Computing sum on GPU..." << std::flush;
  auto result_ct = simple_sum(context, encoder, ui, ct0);
  std::cout << " done" << std::endl;

  std::cout << "[5/6] Decrypting..." << std::flush;
  double result =
      simple_sum__decrypt__result0(context, encoder, ui, result_ct, ui);
  std::cout << " done" << std::endl;

  std::cout << "[6/6] Checking result..." << std::endl;
  std::cout << "  Expected: " << expected << std::endl;
  std::cout << "  Got:      " << result << std::endl;
  std::cout << "  Error:    " << std::abs(result - expected) << std::endl;

  EXPECT_NEAR(result, expected, 0.01)
      << "Sum result doesn't match expected value";

  std::cout << "=== CHEDDAR Simple Sum Test Complete ===" << std::endl;
}
