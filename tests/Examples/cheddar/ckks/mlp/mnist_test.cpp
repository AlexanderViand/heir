// Benchmark: MNIST MLP with split-preprocessing on CHEDDAR GPU.
// Compiles mnist.mlir through HEIR with `split-preprocessing=8`, so the
// plaintext weight encoding is factored into `mnist__preprocessing` and
// the FHE eval is a separate `mnist__preprocessed` call.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
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

TEST(CheddarMNIST, EndToEnd) {
  std::cout << "=== CHEDDAR MNIST Benchmark (split-preprocessing) ==="
            << std::endl;

  std::cout << "[1/5] Configuring CHEDDAR context..." << std::flush;
  auto t0 = sync_now();
  auto [context, ui] = __configure();
  auto& encoder = context->encoder_;
  std::cout << " done (" << ms_between(t0, sync_now()) << " ms)" << std::endl;

  std::cout << "[2/5] Encrypting input..." << std::flush;
  std::vector<double> input_clear(784, 0.5);
  t0 = sync_now();
  auto encrypted_input =
      mnist__encrypt__arg4(context, encoder, ui, input_clear, ui);
  std::cout << " done (" << ms_between(t0, sync_now()) << " ms)" << std::endl;

  // Dummy weights / biases; shapes match mnist.mlir function signature.
  std::vector<double> fc1_weights(512 * 784, 0.01);
  std::vector<double> fc1_bias(512, 0.0);
  std::vector<double> fc2_weights(10 * 512, 0.01);
  std::vector<double> fc2_bias(10, 0.0);

  std::cout << "[3/5] Preprocessing plaintext weights..." << std::flush;
  t0 = sync_now();
  auto [pre0, pre1, pre2, pre3, pre4, pre5, pre6, pre7] = mnist__preprocessing(
      context, encoder, ui, fc1_weights, fc1_bias, fc2_weights, fc2_bias);
  std::cout << " done (" << ms_between(t0, sync_now()) << " ms)" << std::endl;

  // Warmup absorbs one-time costs (CUDA ctx init, allocator, JIT, caches);
  // its result is reused as the sink for the timed loop so the last iteration
  // leaves a valid ciphertext for the decrypt phase.
  std::cout << "[4/5] Running preprocessed MNIST core (warmup + 5 runs)..."
            << std::flush;
  auto result_ct =
      mnist__preprocessed(context, encoder, ui, encrypted_input, pre0, pre1,
                          pre2, pre3, pre4, pre5, pre6, pre7);
  cudaDeviceSynchronize();

  constexpr int kReps = 5;
  std::vector<int64_t> runs_ms;
  runs_ms.reserve(kReps);
  for (int i = 0; i < kReps; ++i) {
    auto s = sync_now();
    result_ct = mnist__preprocessed(context, encoder, ui, encrypted_input, pre0,
                                    pre1, pre2, pre3, pre4, pre5, pre6, pre7);
    runs_ms.push_back(ms_between(s, sync_now()));
  }
  std::sort(runs_ms.begin(), runs_ms.end());
  std::cout << " done (min=" << runs_ms.front()
            << " ms, median=" << runs_ms[kReps / 2]
            << " ms, max=" << runs_ms.back() << " ms)" << std::endl;

  std::cout << "[5/5] Decrypting result..." << std::flush;
  t0 = sync_now();
  auto result = mnist__decrypt__result0(context, encoder, ui, result_ct, ui);
  std::cout << " done (" << ms_between(t0, sync_now()) << " ms)" << std::endl;

  std::cout << "\nOutput (first 10 values):" << std::endl;
  for (int i = 0; i < std::min(10, static_cast<int>(result.size())); ++i) {
    std::cout << "  [" << i << "] = " << result[i] << std::endl;
  }
  std::cout << "\n=== CHEDDAR MNIST Complete ===" << std::endl;
}
