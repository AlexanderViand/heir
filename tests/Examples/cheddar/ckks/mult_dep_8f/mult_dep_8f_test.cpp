// End-to-end mult_dep test using CHEDDAR GPU FHE backend.
// Computes x^8 for a scalar f32 input.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "tests/Examples/cheddar/ckks/mult_dep_8f/mult_dep_lib.h"

using namespace cheddar;
using word = uint64_t;
using Complex = std::complex<double>;

TEST(CheddarMultDep, EndToEnd) {
  std::cout << "=== CHEDDAR Mult Dep 8f Test ===" << std::endl;

  std::cout << "[1/6] Creating context..." << std::flush;
  auto [context, ui] = __configure();
  std::cout << " done" << std::endl;

  std::cout << "[2/6] Keys ready..." << std::endl;

  // Input: x = 0.5, expected: x^8 = 0.00390625
  float x = 0.5;
  double expected = 0.00390625;

  std::cout << "[3/6] Encrypting input..." << std::flush;
  auto& encoder = context->encoder_;
  auto ct0 = mult_dep__encrypt__arg0(context, encoder, ui, x, ui);
  std::cout << " done" << std::endl;

  std::cout << "[4/6] Computing mult_dep on GPU..." << std::flush;
  auto result_ct = mult_dep(context, encoder, ui, ct0);
  std::cout << " done" << std::endl;

  std::cout << "[5/6] Decrypting..." << std::flush;
  double result =
      mult_dep__decrypt__result0(context, encoder, ui, result_ct, ui);
  std::cout << " done" << std::endl;

  std::cout << "[6/6] Checking result..." << std::endl;
  std::cout << "  Expected: " << expected << std::endl;
  std::cout << "  Got:      " << result << std::endl;
  std::cout << "  Error:    " << std::abs(result - expected) << std::endl;

  EXPECT_NEAR(result, expected, 0.001)
      << "mult_dep result doesn't match expected value";

  std::cout << "=== CHEDDAR Mult Dep Test Complete ===" << std::endl;
}
