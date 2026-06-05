// Cheap (no-GPU) helper: construct the orion-MLP CKKS parameter set
// (logN=13, 6 main primes, 2 aux primes, scale 2^26) and print CHEDDAR's exact
// canonical per-level GetScale(level), used to bake the bias-encode scales in
// orion_mlp_lt.mlir (the emitter emits the scale verbatim; CHEDDAR rejects
// mismatches beyond 1e-12).
//
//   bazel run //tests/Examples/cheddar/orion_mlp_lt:scale_dump
//   --//:enable_cheddar=1
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

#include "core/Parameter.h"

using word = uint64_t;

int main() {
  std::vector<word> main_primes = {536903681ULL, 67043329ULL, 66994177ULL,
                                   67239937ULL,  66961409ULL, 66813953ULL};
  std::vector<word> aux_primes = {536952833ULL, 536690689ULL};
  std::vector<word> ter_primes = {};
  std::vector<std::pair<int, int>> level_config;
  for (int i = 1; i <= 6; ++i) level_config.emplace_back(i, 0);
  std::pair<int, int> additional_base = {0, 0};

  cheddar::Parameter<word> p(/*log_degree=*/13,
                             /*base_scale=*/static_cast<double>(1ULL << 26),
                             /*default_encryption_level=*/5, level_config,
                             main_primes, aux_primes, ter_primes,
                             additional_base);

  std::printf("max_level_=%d default_encryption_level_=%d\n", p.max_level_,
              p.default_encryption_level_);
  for (int l = 0; l <= p.default_encryption_level_; ++l) {
    double s = p.GetScale(l);
    uint64_t bits;
    std::memcpy(&bits, &s, sizeof(bits));
    std::printf("GetScale(%d) = %.17g  hex=0x%016llX\n", l, s,
                static_cast<unsigned long long>(bits));
  }
  return 0;
}
