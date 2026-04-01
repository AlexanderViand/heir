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

  // Parameters from the compiled MLIR's cheddar.Q / cheddar.P attributes.
  // Uses min-levels=25 to ensure enough RNS primes for CHEDDAR's GPU.
  std::vector<word> main_primes = {
      36028797014376449ULL, 35184388734977ULL, 35184339320833ULL,
      35184386899969ULL,    35184343908353ULL, 35184385196033ULL,
      35184345088001ULL,    35184383754241ULL, 35184346267649ULL,
      35184383229953ULL,    35184350330881ULL, 35184382967809ULL,
      35184351772673ULL,    35184380870657ULL, 35184353083393ULL,
      35184379035649ULL,    35184355704833ULL, 35184378511361ULL,
      35184358850561ULL,    35184377331713ULL, 35184363569153ULL,
      35184376545281ULL,    35184365273089ULL, 35184373006337ULL,
      35184368025601ULL,    35184372744193ULL};
  std::vector<word> aux_primes = {
      1152921504614055937ULL, 1152921504615628801ULL, 1152921504616808449ULL,
      1152921504618381313ULL, 1152921504620347393ULL, 1152921504622575617ULL,
      1152921504625328129ULL};

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
  // Rotation keys needed by the matvec (from compiled MLIR).
  for (int rot : {1, 2, 3, 4, 8, 12}) {
    ui.PrepareRotationKey(rot, default_encryption_level);
  }
  std::cout << " done" << std::endl;

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
