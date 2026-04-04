// End-to-end dot product test using CHEDDAR GPU FHE backend.
// Mirrors tests/Examples/lattigo/ckks/dot_product_8f/dot_product_8f_test.go

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "tests/Examples/cheddar/ckks/dot_product_8f/dot_product_lib.h"

using namespace cheddar;
using word = uint64_t;
using Complex = std::complex<double>;

TEST(CheddarDotProduct, EndToEnd) {
  std::cout << "=== CHEDDAR Dot Product 8f Test ===" << std::endl;

  std::cout << "[1/6] Creating context..." << std::flush;
  auto [context, ui] = __configure();
  std::cout << " done" << std::endl;

  std::cout << "[2/6] Keys ready..." << std::endl;

  // Input vectors
  std::vector<double> arg0 = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  std::vector<double> arg1 = {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
  double expected = 2.50;

  std::cout << "[3/6] Encrypting inputs..." << std::flush;
  auto& encoder = context->encoder_;
  auto ct0 = dot_product__encrypt__arg0(context, encoder, ui, arg0, ui);
  auto ct1 = dot_product__encrypt__arg1(context, encoder, ui, arg1, ui);
  std::cout << " done" << std::endl;

  std::cout << "[4/6] Computing dot product on GPU..." << std::flush;
  auto result_ct = dot_product(context, encoder, ui, ct0, ct1);
  std::cout << " done" << std::endl;

  std::cout << "[5/6] Decrypting..." << std::flush;
  double result =
      dot_product__decrypt__result0(context, encoder, ui, result_ct, ui);
  std::cout << " done" << std::endl;

  std::cout << "[6/6] Checking result..." << std::endl;
  std::cout << "  Expected: " << expected << std::endl;
  std::cout << "  Got:      " << result << std::endl;
  std::cout << "  Error:    " << std::abs(result - expected) << std::endl;

  EXPECT_NEAR(result, expected, 0.01)
      << "Dot product result doesn't match expected value";

  std::cout << "=== CHEDDAR Dot Product Test Complete ===" << std::endl;
}
