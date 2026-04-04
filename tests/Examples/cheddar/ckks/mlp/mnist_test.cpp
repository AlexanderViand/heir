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

  std::cout << "[1/5] Creating CHEDDAR context..." << std::flush;
  auto start = std::chrono::high_resolution_clock::now();

  auto [context, ui] = __configure();

  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  std::cout
      << " done ("
      << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
      << " ms)" << std::endl;

  std::cout << "[2/5] Keys ready..." << std::endl;

  // 3. Prepare input (all 1s for functional test)
  std::cout << "[3/5] Encrypting input..." << std::flush;
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

  // 4. Prepare weights and plaintext preprocessing.
  std::vector<double> fc1_weights(512 * 784, 0.01);
  std::vector<double> fc1_bias(512, 0.0);
  std::vector<double> fc2_weights(10 * 512, 0.01);
  std::vector<double> fc2_bias(10, 0.0);

  std::cout << "[4/6] Preprocessing plaintext weights..." << std::flush;
  start = std::chrono::high_resolution_clock::now();
  auto [pre0, pre1, pre2, pre3, pre4, pre5, pre6, pre7] = mnist__preprocessing(
      context, encoder, ui, fc1_weights, fc1_bias, fc2_weights, fc2_bias);

  elapsed = std::chrono::high_resolution_clock::now() - start;
  std::cout
      << " done ("
      << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
      << " ms)" << std::endl;

  std::cout << "[5/6] Running preprocessed MNIST core on GPU..." << std::flush;
  start = std::chrono::high_resolution_clock::now();

  auto result_ct =
      mnist__preprocessed(context, encoder, ui, encrypted_input, pre0, pre1,
                          pre2, pre3, pre4, pre5, pre6, pre7);

  elapsed = std::chrono::high_resolution_clock::now() - start;
  std::cout
      << " done ("
      << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
      << " ms)" << std::endl;

  // 6. Decrypt and check
  std::cout << "[6/6] Decrypting result..." << std::flush;
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
