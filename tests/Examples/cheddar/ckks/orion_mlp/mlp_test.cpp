// Benchmark: Orion MLP (BSGS linear_transform + x^2 activation) on CHEDDAR GPU.
// Counterpart to //tests/Examples/orion/mlp:mlp_test (Lattigo, CPU).

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>

#include "tests/Examples/cheddar/ckks/orion_mlp/mlp_lib.h"

using namespace cheddar;
using word = uint64_t;

TEST(CheddarOrionMLP, EndToEnd) {
  std::cout << "=== CHEDDAR Orion MLP Benchmark ===" << std::endl;

  std::cout << "[1/4] Creating CHEDDAR context..." << std::flush;
  auto start = std::chrono::high_resolution_clock::now();
  auto [context, ui] = __configure();
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  std::cout
      << " done ("
      << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
      << " ms)" << std::endl;

  auto& encoder = context->encoder_;

  // Input: 4096-slot ciphertext (matches !ct_L5 in orion/mlp/mlp.mlir).
  std::vector<double> input_clear(4096, 0.5);

  std::cout << "[2/4] Encrypting input..." << std::flush;
  start = std::chrono::high_resolution_clock::now();
  auto ct_in = mlp__encrypt__arg0(context, encoder, ui, input_clear, ui);
  elapsed = std::chrono::high_resolution_clock::now() - start;
  std::cout
      << " done ("
      << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
      << " ms)" << std::endl;

  // Dummy weights / biases: shapes match orion/mlp/mlp.mlir function signature.
  std::vector<double> fc1_w(128 * 4096, 0.01);
  std::vector<double> fc1_b(4096, 0.0);
  std::vector<double> fc2_w(128 * 4096, 0.01);
  std::vector<double> fc2_b(4096, 0.0);
  std::vector<double> fc3_w(137 * 4096, 0.01);
  std::vector<double> fc3_b(4096, 0.0);

  std::cout << "[3/4] Running Orion MLP inference on GPU..." << std::flush;
  start = std::chrono::high_resolution_clock::now();
  auto result_ct = mlp(context, encoder, ui, ct_in, fc1_w, fc1_b, fc2_w, fc2_b,
                       fc3_w, fc3_b);
  elapsed = std::chrono::high_resolution_clock::now() - start;
  std::cout
      << " done ("
      << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
      << " ms)" << std::endl;

  std::cout << "[4/4] Decrypting result..." << std::flush;
  start = std::chrono::high_resolution_clock::now();
  auto result = mlp__decrypt__result0(context, encoder, ui, result_ct, ui);
  elapsed = std::chrono::high_resolution_clock::now() - start;
  std::cout
      << " done ("
      << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
      << " ms)" << std::endl;

  std::cout << "\nOutput (first 10 values):" << std::endl;
  for (int i = 0; i < std::min(10, static_cast<int>(result.size())); ++i) {
    std::cout << "  [" << i << "] = " << result[i] << std::endl;
  }
  std::cout << "\n=== CHEDDAR Orion MLP Complete ===" << std::endl;
}
