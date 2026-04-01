// End-to-end MLP inference test using CHEDDAR GPU FHE backend.
// Tests the HEIR-generated mlp() function with encrypted input on GPU.
//
// This mirrors tests/Examples/orion/mlp/mlp_test.go but targets CHEDDAR.
// Uses the same Orion MLP architecture (3-layer FC with x^2 activation,
// 4096 slots, logN=13, logDefaultScale=26).

#include <gtest/gtest.h>

#include <chrono>
#include <complex>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "cheddar/UserInterface.h"
#include "cheddar/core/Container.h"
#include "cheddar/core/Context.h"
#include "cheddar/core/Encode.h"
#include "cheddar/core/EvkMap.h"
#include "cheddar/core/EvkRequest.h"
#include "cheddar/core/Parameter.h"
#include "cheddar/extension/BootContext.h"
#include "cheddar/extension/LinearTransform.h"
#include "mlp_lib.h"

using namespace cheddar;
using word = uint64_t;
using Complex = std::complex<double>;

// Load CHEDDAR parameters from JSON file.
// Returns a Parameter<word> suitable for the Orion MLP (logN=13, 6 levels).
Parameter<word> loadParams(const std::string& path) {
  // The orion MLP uses logN=13 (N=8192, 4096 slots), 6-level modulus chain,
  // logDefaultScale=26. We need to construct matching 64-bit parameters.
  //
  // Primes from mlp.mlir: Q = [536903681, 67043329, 66994177, 67239937,
  //                             66961409, 66813953]
  //                        P = [536952833, 536690689]
  //
  // For 64-bit CHEDDAR, we use these directly.
  int log_degree = 13;
  uint64_t base_scale = 1ULL << 26;
  int default_encryption_level = 5;

  // Main primes (from the MLIR module's scheme params)
  std::vector<uint64_t> main_primes = {536903681, 67043329, 66994177,
                                       67239937,  66961409, 66813953};
  // Auxiliary primes (for key switching)
  std::vector<uint64_t> aux_primes = {536952833, 536690689};

  // Level config: pairs of (num_main_primes, num_aux_primes) per level
  std::vector<std::pair<int, int>> level_config;
  for (int i = 0; i <= default_encryption_level; ++i) {
    level_config.push_back({static_cast<int>(main_primes.size()) - i, 0});
  }

  return Parameter<word>(log_degree, base_scale, default_encryption_level,
                         level_config, main_primes, aux_primes);
}

TEST(CheddarMLP, EndToEnd) {
  std::cout << "=== CHEDDAR MLP End-to-End Test ===" << std::endl;

  // 1. Setup parameters and context
  std::cout << "[1/6] Creating CHEDDAR context..." << std::flush;
  auto start = std::chrono::high_resolution_clock::now();

  auto param = loadParams("");
  auto context = Context<word>::Create(param);

  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  std::cout
      << " done ("
      << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
      << " ms)" << std::endl;

  // 2. Key generation
  std::cout << "[2/6] Generating keys..." << std::flush;
  start = std::chrono::high_resolution_clock::now();

  UserInterface<word> ui(context);
  // Prepare rotation keys needed by the MLP
  // The Orion MLP uses rotations: 128, 256, 512, 1024, 2048 (for reduce)
  for (int rot : {128, 256, 512, 1024, 2048}) {
    ui.PrepareRotationKey(rot);
  }

  elapsed = std::chrono::high_resolution_clock::now() - start;
  std::cout
      << " done ("
      << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
      << " ms)" << std::endl;

  // 3. Prepare input (all 1s, same as lattigo test)
  int slots = 1 << (param.GetLogDegree() - 1);  // N/2 = 4096
  std::cout << "[3/6] Encoding and encrypting input (" << slots << " slots)..."
            << std::flush;
  start = std::chrono::high_resolution_clock::now();

  std::vector<Complex> input(slots, Complex(1.0, 0.0));
  Plaintext<word> pt_input;
  context->encoder_.Encode(pt_input, param.GetDefaultEncryptionLevel(),
                           static_cast<double>(1ULL << 26), input);
  Ciphertext<word> ct_input;
  ui.Encrypt(ct_input, pt_input);

  elapsed = std::chrono::high_resolution_clock::now() - start;
  std::cout
      << " done ("
      << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
      << " ms)" << std::endl;

  // 4. Prepare weights (all 1s for functional test, same as lattigo test)
  std::cout << "[4/6] Preparing weights..." << std::flush;
  start = std::chrono::high_resolution_clock::now();

  std::vector<Complex> fc1_weights(128 * slots, Complex(1.0, 0.0));
  std::vector<Complex> fc1_bias(slots, Complex(0.0, 0.0));
  std::vector<Complex> fc2_weights(128 * slots, Complex(1.0, 0.0));
  std::vector<Complex> fc2_bias(slots, Complex(0.0, 0.0));
  std::vector<Complex> fc3_weights(137 * slots, Complex(1.0, 0.0));
  std::vector<Complex> fc3_bias(slots, Complex(0.0, 0.0));

  elapsed = std::chrono::high_resolution_clock::now() - start;
  std::cout
      << " done ("
      << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
      << " ms)" << std::endl;

  // 5. Run the MLP
  std::cout << "[5/6] Running MLP inference on GPU..." << std::flush;
  start = std::chrono::high_resolution_clock::now();

  auto& encoder = context->encoder_;
  Ciphertext<word> result_ct =
      mlp(context, encoder, ui, ct_input, fc1_weights, fc1_bias, fc2_weights,
          fc2_bias, fc3_weights, fc3_bias);

  elapsed = std::chrono::high_resolution_clock::now() - start;
  std::cout
      << " done ("
      << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
      << " ms)" << std::endl;

  // 6. Decrypt and check
  std::cout << "[6/6] Decrypting result..." << std::flush;
  start = std::chrono::high_resolution_clock::now();

  Plaintext<word> pt_result;
  ui.Decrypt(pt_result, result_ct);
  std::vector<Complex> result(slots);
  context->encoder_.Decode(result, pt_result);

  elapsed = std::chrono::high_resolution_clock::now() - start;
  std::cout
      << " done ("
      << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
      << " ms)" << std::endl;

  // Print first 10 output values
  std::cout << "\nOutput (first 10 slots):" << std::endl;
  for (int i = 0; i < 10; ++i) {
    std::cout << "  slot[" << i << "] = " << result[i].real() << " + "
              << result[i].imag() << "i" << std::endl;
  }

  // Basic sanity: with all-1s input and weights, values should be nonzero
  bool any_nonzero = false;
  for (int i = 0; i < 10; ++i) {
    if (std::abs(result[i].real()) > 1e-6) {
      any_nonzero = true;
      break;
    }
  }
  EXPECT_TRUE(any_nonzero)
      << "All output values are zero — something went wrong";

  std::cout << "\n=== CHEDDAR MLP Test Complete ===" << std::endl;
}
