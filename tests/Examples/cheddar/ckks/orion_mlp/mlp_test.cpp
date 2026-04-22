// Benchmark: Orion MLP (BSGS linear_transform + x^2 activation) on CHEDDAR GPU.
// Counterpart to //tests/Examples/orion/mlp:mlp_test (Lattigo, CPU).

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>

#include "tests/Examples/cheddar/ckks/orion_mlp/mlp_lib.h"

using namespace cheddar;
using word = uint64_t;

// Sync the GPU and take a wall-clock sample; CHEDDAR launches kernels async,
// so without the sync we'd be timing launches, not completions.
static std::chrono::high_resolution_clock::time_point sync_now() {
  cudaDeviceSynchronize();
  return std::chrono::high_resolution_clock::now();
}

static int64_t ms_between(std::chrono::high_resolution_clock::time_point a,
                          std::chrono::high_resolution_clock::time_point b) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
}

TEST(CheddarOrionMLP, EndToEnd) {
  std::cout << "=== CHEDDAR Orion MLP Benchmark ===" << std::endl;

  std::cout << "[1/4] Creating CHEDDAR context..." << std::flush;
  auto t0 = sync_now();
  auto [context, ui] = __configure();
  std::cout << " done (" << ms_between(t0, sync_now()) << " ms)" << std::endl;

  auto& encoder = context->encoder_;

  // Input: 4096-slot ciphertext (matches !ct_L5 in orion/mlp/mlp.mlir).
  std::vector<double> input_clear(4096, 0.5);

  std::cout << "[2/4] Encrypting input..." << std::flush;
  t0 = sync_now();
  auto ct_in = mlp__encrypt__arg0(context, encoder, ui, input_clear, ui);
  std::cout << " done (" << ms_between(t0, sync_now()) << " ms)" << std::endl;

  // Dummy weights / biases: shapes match orion/mlp/mlp.mlir function signature.
  std::vector<double> fc1_w(128 * 4096, 0.01);
  std::vector<double> fc1_b(4096, 0.0);
  std::vector<double> fc2_w(128 * 4096, 0.01);
  std::vector<double> fc2_b(4096, 0.0);
  std::vector<double> fc3_w(137 * 4096, 0.01);
  std::vector<double> fc3_b(4096, 0.0);

  // Warmup absorbs one-time costs (CUDA ctx init, allocator, JIT, caches);
  // its result is reused as the sink for the timed loop so the last iteration
  // leaves a valid ciphertext for the decrypt phase.
  std::cout << "[3/4] Running Orion MLP inference (warmup + 5 runs)..."
            << std::flush;
  auto result_ct = mlp(context, encoder, ui, ct_in, fc1_w, fc1_b, fc2_w, fc2_b,
                       fc3_w, fc3_b);
  cudaDeviceSynchronize();

  constexpr int kReps = 5;
  std::vector<int64_t> runs_ms;
  runs_ms.reserve(kReps);
  for (int i = 0; i < kReps; ++i) {
    auto s = sync_now();
    result_ct = mlp(context, encoder, ui, ct_in, fc1_w, fc1_b, fc2_w, fc2_b,
                    fc3_w, fc3_b);
    runs_ms.push_back(ms_between(s, sync_now()));
  }
  std::sort(runs_ms.begin(), runs_ms.end());
  std::cout << " done (min=" << runs_ms.front()
            << " ms, median=" << runs_ms[kReps / 2]
            << " ms, max=" << runs_ms.back() << " ms)" << std::endl;

  std::cout << "[4/4] Decrypting result..." << std::flush;
  t0 = sync_now();
  auto result = mlp__decrypt__result0(context, encoder, ui, result_ct, ui);
  std::cout << " done (" << ms_between(t0, sync_now()) << " ms)" << std::endl;

  std::cout << "\nOutput (first 10 values):" << std::endl;
  for (int i = 0; i < std::min(10, static_cast<int>(result.size())); ++i) {
    std::cout << "  [" << i << "] = " << result[i] << std::endl;
  }
  std::cout << "\n=== CHEDDAR Orion MLP Complete ===" << std::endl;
}
