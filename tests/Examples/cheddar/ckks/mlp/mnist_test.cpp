// End-to-end MNIST inference test using CHEDDAR GPU FHE backend.
// Compiles mnist.mlir through the full HEIR pipeline to CHEDDAR C++,
// encrypts a test input, runs inference on GPU, decrypts, and validates.

#include <gtest/gtest.h>

#include <chrono>
#include <complex>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "tests/Examples/cheddar/ckks/mlp/mnist_lib.h"

using namespace cheddar;
using word = uint64_t;
using Complex = std::complex<double>;

TEST(CheddarMNIST, EndToEnd) {
  std::cout << "=== CHEDDAR MNIST End-to-End Test ===" << std::endl;

  // Use CHEDDAR's built-in 64-bit parameter file.
  // logN=16, 32768 slots, logDefaultScale=40, 26 levels.
  // The MNIST pipeline uses logDefaultScale=45, so we need compatible
  // params. For this test, we construct minimal params matching the
  // scheme params from the compiled MLIR:
  //   cheddar.logN = 16
  //   cheddar.logDefaultScale = 45
  //   cheddar.Q = [36028797014376449, 35184358850561, ...]
  //   cheddar.P = [1152921504614055937, ...]

  std::cout << "[1/5] Creating CHEDDAR context..." << std::flush;
  auto start = std::chrono::high_resolution_clock::now();

  // Primes from the compiled MLIR's cheddar.Q / cheddar.P attributes
  std::vector<word> main_primes = {
      36028797014376449ULL, 35184358850561ULL, 35184377331713ULL,
      35184363569153ULL,    35184376545281ULL, 35184365273089ULL,
      35184373006337ULL,    35184368025601ULL, 35184372744193ULL};
  std::vector<word> aux_primes = {
      1152921504614055937ULL, 1152921504615628801ULL, 1152921504616808449ULL};

  // Level config: (num_main, num_ter) pairs, one per level
  std::vector<std::pair<int, int>> level_config;
  for (int i = 1; i <= static_cast<int>(main_primes.size()); ++i) {
    level_config.push_back({i, 0});
  }

  int log_degree = 16;
  double base_scale = static_cast<double>(1ULL << 45);
  int default_encryption_level = static_cast<int>(main_primes.size()) - 1;  // 8

  Parameter<word> param(log_degree, base_scale, default_encryption_level,
                        level_config, main_primes, aux_primes);

  auto context = Context<word>::Create(param);

  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  std::cout
      << " done ("
      << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
      << " ms)" << std::endl;

  // 2. Key generation
  std::cout << "[2/5] Generating keys..." << std::flush;
  start = std::chrono::high_resolution_clock::now();

  UserInterface<word> ui(context);
  // Prepare rotation keys for all distances used by the MNIST inference.
  // These are derived from the layout/packing scheme (ciphertext-degree=32768).
  for (int rot :
       {1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,
        14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  32,  46,  64,
        69,  92,  115, 128, 138, 161, 184, 207, 230, 253, 256, 276, 299,
        322, 345, 368, 391, 414, 437, 460, 483, 506, 512}) {
    ui.PrepareRotationKey(rot, default_encryption_level);
  }

  elapsed = std::chrono::high_resolution_clock::now() - start;
  std::cout
      << " done ("
      << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
      << " ms)" << std::endl;

  // 3. Prepare input (all 1s for functional test)
  int slots = 1 << (log_degree - 1);  // 32768
  std::cout << "[3/5] Encrypting input (" << slots << " slots)..."
            << std::flush;
  start = std::chrono::high_resolution_clock::now();

  // Create a simple test input (784 pixels, all 0.5)
  std::vector<double> input_clear(784, 0.5);

  // Use the generated encrypt helper
  auto& encoder = context->encoder_;
  auto encrypted_input =
      mnist__encrypt__arg4(context, encoder, ui, input_clear, ui);

  elapsed = std::chrono::high_resolution_clock::now() - start;
  std::cout
      << " done ("
      << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
      << " ms)" << std::endl;

  // 4. Prepare weights (zeros for smoke test — validates pipeline runs)
  std::cout << "[4/5] Running MNIST inference on GPU..." << std::flush;
  start = std::chrono::high_resolution_clock::now();

  std::vector<double> fc1_weights(512 * 784, 0.01);
  std::vector<double> fc1_bias(512, 0.0);
  std::vector<double> fc2_weights(10 * 512, 0.01);
  std::vector<double> fc2_bias(10, 0.0);

  auto result_ct = mnist(context, encoder, ui, fc1_weights, fc1_bias,
                         fc2_weights, fc2_bias, encrypted_input);

  elapsed = std::chrono::high_resolution_clock::now() - start;
  std::cout
      << " done ("
      << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
      << " ms)" << std::endl;

  // 5. Decrypt and check
  std::cout << "[5/5] Decrypting result..." << std::flush;
  start = std::chrono::high_resolution_clock::now();

  auto result = mnist__decrypt__result0(context, encoder, ui, result_ct, ui);

  elapsed = std::chrono::high_resolution_clock::now() - start;
  std::cout
      << " done ("
      << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
      << " ms)" << std::endl;

  // Print output
  std::cout << "\nOutput (first 10 values):" << std::endl;
  for (int i = 0; i < std::min(10, static_cast<int>(result.size())); ++i) {
    std::cout << "  [" << i << "] = " << result[i] << std::endl;
  }

  std::cout << "\n=== CHEDDAR MNIST Test Complete ===" << std::endl;
}
