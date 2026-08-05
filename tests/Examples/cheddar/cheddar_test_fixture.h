// Shared test fixture for CHEDDAR end-to-end tests. Header-only.
//
// All e2e tests need the same boilerplate: build a CKKS Parameter, create a
// Context + UserInterface, then encode/encrypt inputs and decrypt/decode
// outputs. This header bundles that into a `CkksFixture` struct plus a few
// helpers, so individual tests focus on the specific kernel being exercised.

#pragma once

#include <complex>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "UserInterface.h"
#include "core/Context.h"
#include "core/Encode.h"
#include "core/Parameter.h"

namespace cheddar_test {

using word = uint64_t;
using Complex = std::complex<double>;
using Ct = cheddar::Ciphertext<word>;
using Pt = cheddar::Plaintext<word>;
using Const = cheddar::Constant<word>;
using Evk = cheddar::EvaluationKey<word>;
using CtxPtr = std::shared_ptr<cheddar::Context<word>>;
using UI = cheddar::UserInterface<word>;

struct CkksFixture {
  CtxPtr ctx;
  std::unique_ptr<UI> ui;
  int top_level;
};

// Five-level CKKS params, scale = 2^36. Sufficient for kernels that consume
// up to 4 levels of multiplicative depth. Bumps to taste in individual tests.
inline CkksFixture MakeFixture() {
  std::vector<word> main_primes = {
      36028797017456641ULL, 35184366911489ULL, 35184376545281ULL,
      35184367828993ULL,    35184373989377ULL,
  };
  std::vector<word> aux_primes = {
      1152921504608747521ULL,
      1152921504614055937ULL,
  };
  std::vector<std::pair<int, int>> level_config;
  for (int i = 1; i <= static_cast<int>(main_primes.size()); ++i) {
    level_config.emplace_back(i, 0);
  }
  int max_level = static_cast<int>(main_primes.size()) - 1;
  static cheddar::Parameter<word> param(
      /*logN=*/13, /*scale=*/static_cast<double>(1ULL << 36), max_level,
      level_config, main_primes, aux_primes);
  CkksFixture f;
  f.ctx = cheddar::Context<word>::Create(param);
  f.ui = std::make_unique<UI>(f.ctx);
  f.top_level = max_level;
  return f;
}

inline Pt EncodeVec(const CkksFixture& f, const std::vector<double>& msg,
                    int level) {
  std::vector<Complex> complex_msg(msg.begin(), msg.end());
  Pt pt;
  f.ctx->encoder_.Encode(pt, level, f.ctx->param_.GetScale(level), complex_msg);
  return pt;
}

inline Ct EncryptVec(const CkksFixture& f, const std::vector<double>& msg,
                     int level) {
  Pt pt = EncodeVec(f, msg, level);
  Ct ct;
  f.ui->Encrypt(ct, pt);
  return ct;
}

inline std::vector<double> DecryptVec(const CkksFixture& f, const Ct& ct,
                                      size_t n) {
  Pt pt;
  f.ui->Decrypt(pt, ct);
  std::vector<Complex> out;
  f.ctx->encoder_.Decode(out, pt);
  std::vector<double> real_out(n);
  for (size_t i = 0; i < n; ++i) real_out[i] = out[i].real();
  return real_out;
}

}  // namespace cheddar_test
