// End-to-end mult_indep test using CHEDDAR GPU FHE backend.
// Computes the product of 8 independent scalar f32 inputs.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "tests/Examples/cheddar/ckks/mult_indep_8f/mult_indep_lib.h"

using namespace cheddar;
using word = uint64_t;
using Complex = std::complex<double>;

TEST(CheddarMultIndep, EndToEnd) {
  std::cout << "=== CHEDDAR Mult Indep 8f Test ===" << std::endl;

  std::cout << "[1/6] Creating context..." << std::flush;
  auto [context, ui] = __configure();
  std::cout << " done" << std::endl;

  std::cout << "[2/6] Keys ready..." << std::endl;

  // All inputs = 0.5, expected: 0.5^8 = 0.00390625
  float arg0 = 0.5, arg1 = 0.5, arg2 = 0.5, arg3 = 0.5;
  float arg4 = 0.5, arg5 = 0.5, arg6 = 0.5, arg7 = 0.5;
  double expected = 0.00390625;

  std::cout << "[3/6] Encrypting inputs..." << std::flush;
  auto& encoder = context->encoder_;
  auto ct0 = mult_indep__encrypt__arg0(context, encoder, ui, arg0, ui);
  auto ct1 = mult_indep__encrypt__arg1(context, encoder, ui, arg1, ui);
  auto ct2 = mult_indep__encrypt__arg2(context, encoder, ui, arg2, ui);
  auto ct3 = mult_indep__encrypt__arg3(context, encoder, ui, arg3, ui);
  auto ct4 = mult_indep__encrypt__arg4(context, encoder, ui, arg4, ui);
  auto ct5 = mult_indep__encrypt__arg5(context, encoder, ui, arg5, ui);
  auto ct6 = mult_indep__encrypt__arg6(context, encoder, ui, arg6, ui);
  auto ct7 = mult_indep__encrypt__arg7(context, encoder, ui, arg7, ui);
  std::cout << " done" << std::endl;

  std::cout << "[4/6] Computing mult_indep on GPU..." << std::flush;
  auto result_ct =
      mult_indep(context, encoder, ui, ct0, ct1, ct2, ct3, ct4, ct5, ct6, ct7);
  std::cout << " done" << std::endl;

  std::cout << "[5/6] Decrypting..." << std::flush;
  double result =
      mult_indep__decrypt__result0(context, encoder, ui, result_ct, ui);
  std::cout << " done" << std::endl;

  std::cout << "[6/6] Checking result..." << std::endl;
  std::cout << "  Expected: " << expected << std::endl;
  std::cout << "  Got:      " << result << std::endl;
  std::cout << "  Error:    " << std::abs(result - expected) << std::endl;

  EXPECT_NEAR(result, expected, 0.001)
      << "mult_indep result doesn't match expected value";

  std::cout << "=== CHEDDAR Mult Indep Test Complete ===" << std::endl;
}
