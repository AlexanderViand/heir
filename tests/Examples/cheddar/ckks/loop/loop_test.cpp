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

  // Parameters from the compiled MLIR's cheddar.Q / cheddar.P attributes.
  // Q = [36028797014376449, 35184377331713, 35184363569153,
  //      35184376545281, 35184365273089, 35184373006337,
  //      35184368025601, 35184372744193]
  // P = [1152921504614055937, 1152921504615628801, 1152921504616808449]
  std::vector<word> main_primes = {36028797014376449ULL, 35184377331713ULL,
                                   35184363569153ULL,    35184376545281ULL,
                                   35184365273089ULL,    35184373006337ULL,
                                   35184368025601ULL,    35184372744193ULL};
  std::vector<word> aux_primes = {
      1152921504614055937ULL, 1152921504615628801ULL, 1152921504616808449ULL};

  std::vector<std::pair<int, int>> level_config;
  for (int i = 1; i <= static_cast<int>(main_primes.size()); ++i) {
    level_config.push_back({i, 0});
  }

  int log_degree = 16;
  double base_scale = static_cast<double>(1ULL << 45);
  int default_encryption_level = static_cast<int>(main_primes.size()) - 1;

  std::cout << "[1/6] Creating context..." << std::flush;
  Parameter<word> param(log_degree, base_scale, default_encryption_level,
                        level_config, main_primes, aux_primes);
  auto context = Context<word>::Create(param);
  std::cout << " done" << std::endl;

  std::cout << "[2/6] Generating keys..." << std::flush;
  UserInterface<word> ui(context);
  // No rotation keys needed for the loop function.
  std::cout << " done" << std::endl;

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
