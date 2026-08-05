// Cheap (no-GPU) helper: construct the bootparam_40_64bit CKKS parameter set
// and print CHEDDAR's exact canonical per-level scale GetScale(level). The
// cheddar emitter emits the encode `scale` attribute verbatim and CHEDDAR
// rejects mismatches beyond 1e-12, so the bootstrap IR's level-0 encode must
// carry the exact GetScale(0); this dumps it (decimal + f64 hex) to bake into
// bootstrap.mlir.
//
//   bazel run //tests/Examples/cheddar/bootstrap:scale_dump
//   --//:enable_cheddar=1
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

#include "core/Parameter.h"

using word = uint64_t;

int main() {
  std::vector<word> main_primes = {
      1125899908022273ULL,  1099515691009ULL,     1099523555329ULL,
      1099525128193ULL,     1099526176769ULL,     1099529060353ULL,
      1099535220737ULL,     1099536138241ULL,     1099537580033ULL,
      1099538104321ULL,     1099540725761ULL,     1099540856833ULL,
      1099543085057ULL,     36028797019488257ULL, 36028797023420417ULL,
      36028797024206849ULL, 36028797025124353ULL, 36028797032202241ULL,
      36028797033644033ULL, 36028797037576193ULL, 36028797048324097ULL,
      36028797048586241ULL, 36028797049896961ULL, 36028797051863041ULL,
      36028797053698049ULL, 36028797054222337ULL};
  std::vector<word> aux_primes = {72057594038321153ULL, 72057594040680449ULL,
                                  72057594042646529ULL, 72057594047889409ULL,
                                  72057594057195521ULL, 72057594058375169ULL,
                                  72057594058899457ULL};
  std::vector<word> ter_primes = {};
  std::vector<std::pair<int, int>> level_config;
  for (int i = 1; i <= 26; ++i) level_config.emplace_back(i, 0);
  std::pair<int, int> additional_base = {0, 0};

  cheddar::Parameter<word> p(/*log_degree=*/16,
                             /*base_scale=*/static_cast<double>(1ULL << 40),
                             /*default_encryption_level=*/13, level_config,
                             main_primes, aux_primes, ter_primes,
                             additional_base);
  p.SetDenseHammingWeight(32768);
  p.SetSparseHammingWeight(32);

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
