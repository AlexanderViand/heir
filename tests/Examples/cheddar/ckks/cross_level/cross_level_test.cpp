// End-to-end cross-level test using CHEDDAR GPU FHE backend.
// Mirrors tests/Examples/lattigo/ckks/cross_level/cross_level_debug_test.go
//
// The computation is:
//   base0 = base + add
//   mul1  = base0 * base0
//   base1 = mul1 + add
//   mul2  = base1 * base1
//   base2 = mul2 + add
//   mul3  = base2 * base2
//   base3 = mul3 + add
//
// With base = {1, 0, 1, 0} and add = {0, 1, 0, 1}:
//   base0 = {1, 1, 1, 1}
//   mul1  = {1, 1, 1, 1}
//   base1 = {1, 2, 1, 2}
//   mul2  = {1, 4, 1, 4}
//   base2 = {1, 5, 1, 5}
//   mul3  = {1, 25, 1, 25}
//   base3 = {1, 26, 1, 26}

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "tests/Examples/cheddar/ckks/cross_level/cross_level_lib.h"

using namespace cheddar;
using word = uint64_t;
using Complex = std::complex<double>;

TEST(CheddarCrossLevel, EndToEnd) {
  std::cout << "=== CHEDDAR Cross Level Test ===" << std::endl;

  std::cout << "[1/6] Creating context..." << std::flush;
  auto [context, ui] = __configure();
  std::cout << " done" << std::endl;

  std::cout << "[2/6] Keys ready..." << std::endl;

  std::vector<double> arg0 = {1, 0, 1, 0};
  std::vector<double> arg1 = {0, 1, 0, 1};
  std::vector<double> expected = {1, 26, 1, 26};

  std::cout << "[3/6] Encrypting inputs..." << std::flush;
  auto& encoder = context->encoder_;
  auto ct0 = cross_level__encrypt__arg0(context, encoder, ui, arg0, ui);
  auto ct1 = cross_level__encrypt__arg1(context, encoder, ui, arg1, ui);
  std::cout << " done" << std::endl;

  std::cout << "[4/6] Computing cross-level ops on GPU..." << std::flush;
  auto result_ct = cross_level(context, encoder, ui, ct0, ct1);
  std::cout << " done" << std::endl;

  std::cout << "[5/6] Decrypting..." << std::flush;
  std::vector<double> result =
      cross_level__decrypt__result0(context, encoder, ui, result_ct, ui);
  std::cout << " done" << std::endl;

  std::cout << "[6/6] Checking results..." << std::endl;
  for (size_t i = 0; i < expected.size(); ++i) {
    std::cout << "  slot " << i << ": expected=" << expected[i]
              << "  got=" << result[i]
              << "  error=" << std::abs(result[i] - expected[i]) << std::endl;
    EXPECT_NEAR(result[i], expected[i], 0.1) << "Mismatch at slot " << i;
  }

  std::cout << "=== CHEDDAR Cross Level Test Complete ===" << std::endl;
}
