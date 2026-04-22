// End-to-end loop test using CHEDDAR GPU FHE backend.
// Computes f(x) = (((...((1*x - 1)*x - 1)*x - 1)...) - 1) with 8 iterations.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "tests/Examples/cheddar/ckks/loop/loop_lib.h"

using namespace cheddar;
using word = uint64_t;
using Complex = std::complex<double>;

TEST(CheddarLoop, EndToEnd) {
  std::cout << "=== CHEDDAR Loop Test ===" << std::endl;

  std::cout << "[1/6] Creating context..." << std::flush;
  auto [context, ui] = __configure();
  std::cout << " done" << std::endl;

  std::cout << "[2/6] Keys ready..." << std::endl;

  // Input: 8 evenly spaced values in [0, 1]
  std::vector<double> arg0 = {0.0,        0.14285714, 0.28571429, 0.42857143,
                              0.57142857, 0.71428571, 0.85714286, 1.0};
  // Expected output of f(x) = (((...((1*x - 1)*x - 1)*x - 1)...) - 1)
  // with 8 iterations.
  std::vector<double> expected = {-1.0,        -1.16666629, -1.39989342,
                                  -1.74687019, -2.29543899, -3.19507837,
                                  -4.66914279, -7.0};

  std::cout << "[3/6] Encrypting input..." << std::flush;
  auto& encoder = context->encoder_;
  auto ct0 = loop__encrypt__arg0(context, encoder, ui, arg0, ui);
  std::cout << " done" << std::endl;

  std::cout << "[4/6] Computing loop on GPU..." << std::flush;
  auto result_ct = loop(context, encoder, ui, ct0);
  std::cout << " done" << std::endl;

  std::cout << "[5/6] Decrypting..." << std::flush;
  std::vector<double> result =
      loop__decrypt__result0(context, encoder, ui, result_ct, ui);
  std::cout << " done" << std::endl;

  std::cout << "[6/6] Checking results..." << std::endl;
  for (size_t i = 0; i < expected.size(); ++i) {
    std::cout << "  slot " << i << ": expected=" << expected[i]
              << "  got=" << result[i]
              << "  error=" << std::abs(result[i] - expected[i]) << std::endl;
    EXPECT_NEAR(result[i], expected[i], 0.05) << "Mismatch at slot " << i;
  }

  std::cout << "=== CHEDDAR Loop Test Complete ===" << std::endl;
}
