// Shared prelude for the CHEDDAR e2e examples' generated kernels.
//
// cheddar-to-emitc lowers cheddar.linear_transform and cheddar.eval_poly to
// calls into these thin HEIR-side shims (CHEDDAR's LinearTransform / EvalPoly
// are classes, not Context methods). Each example's generated `*_raw.cc` is
// concatenated after a one-line `#include` of this header (see the genrules in
// the per-example BUILD files), so this is the single source of truth for the
// shim ABI rather than a copy pasted into every genrule.

#ifndef TESTS_EXAMPLES_CHEDDAR_CHEDDAR_EMIT_PRELUDE_H_
#define TESTS_EXAMPLES_CHEDDAR_CHEDDAR_EMIT_PRELUDE_H_

#include <complex>
#include <cstdint>
#include <initializer_list>
#include <vector>

#include "UserInterface.h"
#include "core/Context.h"
#include "core/Encode.h"
#include "core/Parameter.h"
#include "extension/BootContext.h"  // cheddar.boot lowers to BootContext::Boot
#include "extension/EvalPoly.h"
#include "extension/LinearTransform.h"
#include "extension/StripedMatrix.h"

namespace cheddar {

// The RunLinearTransform shim is emitted into the generated kernel itself
// by cheddar-emitc-boundary (self-contained output); this prelude only
// provides the backend headers it needs.

// cheddar.eval_poly shim: evaluate a Chebyshev series on [-1, 1], rescaling the
// output to the canonical scale of output_level.
template <typename wordT>
void RunEvalPoly(Ciphertext<wordT>& out, Context<wordT>* ctx,
                 const Ciphertext<wordT>& in, const EvkMap<wordT>& evk_map,
                 std::initializer_list<double> coeffs, int level,
                 int output_level) {
  ConstContextPtr<wordT> cp(ConstContextPtr<wordT>(), ctx);
  std::vector<double> c(coeffs);
  EvalPoly<wordT> poly(c, level, ctx->param_.GetScale(level),
                       ctx->param_.GetScale(output_level), /*chebyshev=*/true);
  poly.Compile(cp);
  poly.Evaluate(cp, out, in, evk_map.GetMultiplicationKey());
}

}  // namespace cheddar

#endif  // TESTS_EXAMPLES_CHEDDAR_CHEDDAR_EMIT_PRELUDE_H_
