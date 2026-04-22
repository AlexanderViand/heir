#include "lib/Parameters/RLWESecurityParams.h"

#include <cassert>

namespace mlir {
namespace heir {

// "Security Guidelines for Implementing Homomorphic Encryption"
// https://ia.cr/2024/463

// 128-bit classic security for uniform ternary secret distribution
struct RLWESecurityParam rlweSecurityParam128BitClassic[] = {
    {1024, 26},   {2048, 53},   {4096, 106},   {8192, 214},
    {16384, 430}, {32768, 868}, {65536, 1747}, {131072, 3523}};

// OpenFHE's HEStd_ternary bounds for 128-bit classic security.
// These are slightly more generous than the standard HE security guidelines
// above, and must be used when targeting OpenFHE to ensure generated
// parameters pass OpenFHE's runtime security validation.
// Source: openfhe-development/src/core/lib/lattice/stdlatticeparms.cpp
struct RLWESecurityParam rlweSecurityParamOpenFHETernary[] = {
    {1024, 27},   {2048, 54},   {4096, 109},   {8192, 218},
    {16384, 438}, {32768, 881}, {65536, 1747}, {131072, 3523}};

int computeRingDim(int logPQ, int minRingDim, bool useOpenFHEBounds) {
  auto& table = useOpenFHEBounds ? rlweSecurityParamOpenFHETernary
                                 : rlweSecurityParam128BitClassic;
  for (auto& param : table) {
    if (param.ringDim < minRingDim) {
      continue;
    }
    if (param.logMaxQ >= logPQ) {
      return param.ringDim;
    }
  }
  assert(false && "Failed to find ring dimension, logTotalPQ too large");
  return 0;
}

}  // namespace heir
}  // namespace mlir
