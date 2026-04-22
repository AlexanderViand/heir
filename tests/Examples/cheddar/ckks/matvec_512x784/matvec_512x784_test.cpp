// End-to-end matvec 512x784 test using CHEDDAR GPU FHE backend.
// Mirrors tests/Examples/lattigo/ckks/matvec_512x784/matvec_512x784_test.go

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "tests/Examples/cheddar/ckks/matvec_512x784/matvec_512x784_lib.h"

using namespace cheddar;
using word = uint64_t;
using Complex = std::complex<double>;

TEST(CheddarMatvec512x784, EndToEnd) {
  std::cout << "=== CHEDDAR Matvec 512x784 Test ===" << std::endl;

  std::cout << "[1/6] Creating context..." << std::flush;
  auto [context, ui] = __configure();
  std::cout << " done" << std::endl;

  std::cout << "[2/6] Keys ready..." << std::endl;

  // Input vector: 784 values all set to 0.1
  int cols = 784;
  int rows = 512;
  std::vector<double> arg0(cols, 0.1);

  // Matrix is all 1.0, so each output row = sum of input = 784 * 0.1 = 78.4
  double expected = 78.4;

  std::cout << "[3/6] Encrypting input..." << std::flush;
  auto& encoder = context->encoder_;
  auto ct0 = matvec__encrypt__arg0(context, encoder, ui, arg0, ui);
  std::cout << " done" << std::endl;

  std::cout << "[4/6] Computing matvec on GPU..." << std::flush;
  auto result_ct = matvec(context, encoder, ui, ct0);
  std::cout << " done" << std::endl;

  std::cout << "[5/6] Decrypting..." << std::flush;
  std::vector<double> result =
      matvec__decrypt__result0(context, encoder, ui, result_ct, ui);
  std::cout << " done" << std::endl;

  std::cout << "[6/6] Checking results..." << std::endl;
  double errorThreshold = 2.0;
  for (int i = 0; i < rows; ++i) {
    if (i < 10 || std::abs(result[i] - expected) > errorThreshold) {
      std::cout << "  row " << i << ": expected=" << expected
                << "  got=" << result[i]
                << "  error=" << std::abs(result[i] - expected) << std::endl;
    }
    EXPECT_NEAR(result[i], expected, errorThreshold) << "Mismatch at row " << i;
  }

  std::cout << "=== CHEDDAR Matvec 512x784 Test Complete ===" << std::endl;
}
