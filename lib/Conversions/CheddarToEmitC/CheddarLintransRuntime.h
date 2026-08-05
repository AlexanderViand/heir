// The CHEDDAR linear-transform runtime support that cheddar-emitc-boundary
// injects verbatim into generated kernels (via the CheddarLintransRuntime.inc
// genrule), so each generated TU stays self-contained. Checked in as a real
// header so it is reviewable and compilable as C++: the compile-only test
// under tests/Conversions/CheddarToEmitC/compile builds an emitted kernel
// containing this code against a header-only CHEDDAR stub, which pins both
// the API contract and the template instantiations.
//
// Consumer contract: the including TU must provide the CHEDDAR headers
// (extension/linalg/{LinearTransform,StripedMatrix}.h or the test stub),
// `using word = ...;` and the cheddar namespace BEFORE this code; only std
// includes and fully qualified names are used here.
//
// NB: the injected copy is guarded by the same include guard, so a consumer
// that also #includes this header directly does not redefine anything.
#ifndef LIB_CONVERSIONS_CHEDDARTOEMITC_CHEDDARLINTRANSRUNTIME_H_
#define LIB_CONVERSIONS_CHEDDARTOEMITC_CHEDDARLINTRANSRUNTIME_H_

// cheddar.linear_transform shim (emitted by cheddar-emitc-boundary): pack the
// diagonal-packed weights into a StripedMatrix, construct the transform at
// the op's level/scale, evaluate with CHEDDAR's hoisted BSGS kernel. diagT:
// f64 (orion frontend) or f32 (torch-linalg path). Construction encodes every
// diagonal (the dominant cost); the weights are process-lifetime constants,
// so the transform is memoized on the diagonal array's address plus the
// construction parameters (level, bs, gs).
#include <algorithm>
#include <complex>
#include <initializer_list>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <vector>
// MinKS (minimal key-switching) eligibility. CHEDDAR's Evaluate /
// AddRequiredRotations assert (abort) with min_ks=true unless the transform's
// nonzero baby AND giant rotations each form a COMPLETE arithmetic
// progression. IsUsingBSGS() alone (bs>1 && gs>1) is too permissive: a
// gap-structured transform (e.g. a dilated conv1d's expanded Toeplitz)
// satisfies it yet fails the progression check -> std::exit(1). Compute the
// check on the diagonal map once at construction, memoize per transform, and
// use it identically for keygen and eval so a needed key is never missing.
static int LinearTransformGCD(int a, int b) {
  while (b != 0) {
    int r = a % b;
    a = b;
    b = r;
  }
  return a < 0 ? -a : a;
}
static std::map<const void*, bool>& LinearTransformMinKSMap() {
  static auto* modes = new std::map<const void*, bool>();
  return *modes;
}
template <typename Matrix>
static bool CanUseLinearTransformMinKS(const Matrix& matrix, int W, int bs,
                                       int gs) {
  if (bs <= 1 || gs <= 1) return false;
  int stride = 0;
  for (const auto& item : matrix) {
    int rot = item.first % W;
    if (rot < 0) rot += W;
    stride = LinearTransformGCD(stride, rot);
  }
  if (stride == 0) return false;
  int gs_stride = stride * bs;
  std::set<int> bs_indices;
  std::set<int> gs_indices;
  for (const auto& item : matrix) {
    int rot = item.first % W;
    if (rot < 0) rot += W;
    int bs_rot = rot % gs_stride;
    bs_indices.insert(bs_rot);
    gs_indices.insert(rot - bs_rot);
  }
  auto is_complete_stride = [](const std::set<int>& indices) {
    int count = 0;
    int gcd = 0;
    int maximum = 0;
    for (int index : indices) {
      if (index == 0) continue;
      gcd = LinearTransformGCD(gcd, index);
      maximum = std::max(maximum, index);
      ++count;
    }
    return count > 0 && count * gcd == maximum;
  };
  return is_complete_stride(bs_indices) && is_complete_stride(gs_indices);
}
static bool LinearTransformUsesMinKS(const void* transform) {
  auto& modes = LinearTransformMinKSMap();
  auto it = modes.find(transform);
  return it != modes.end() && it->second;
}
template <int W, typename wordT, typename diagT>
static void PrepareLinearTransform(
    std::shared_ptr<cheddar::LinearTransform<wordT>>& out,
    cheddar::Context<wordT>* ctx, diagT diag[][W],
    std::initializer_list<int> idx, int level, int bs, int gs) {
  cheddar::StripedMatrix m(W, W);
  int d = 0;
  for (int k : idx) {
    m[k] = std::vector<std::complex<double>>(diag[d], diag[d] + W);
    ++d;
  }
  cheddar::ConstContextPtr<wordT> cp(cheddar::ConstContextPtr<wordT>(), ctx);
  out = std::make_shared<cheddar::LinearTransform<wordT>>(
      cp, m, level, ctx->param_.GetScale(level), bs, gs);
  // Memoize MinKS eligibility keyed on the transform pointer; keygen and eval
  // both look it up.
  LinearTransformMinKSMap()[out.get()] =
      CanUseLinearTransformMinKS(m, W, bs, gs);
}
// Deferred linear-transform keygen: ask a prepared transform for exactly the
// rotations it needs (post zero-diagonal pruning) so the harness can generate
// only those keys, instead of __configure emitting the full conservative BSGS
// set. AddRequiredRotations takes (req, min_ks) and must use the SAME min_ks
// decision as evaluation, or keygen and eval disagree and a needed key is
// missing; IsUsingBSGS() would over-approximate and abort on gap-structured
// transforms.
template <typename T, typename Req>
static void AddLintransRequiredRotations(const std::shared_ptr<T>& lt,
                                         Req& req) {
  lt->AddRequiredRotations(req, LinearTransformUsesMinKS(lt.get()));
}
template <typename wordT>
static void RunPreparedLinearTransform(
    cheddar::Ciphertext<wordT>& out, cheddar::Context<wordT>* ctx,
    const cheddar::Ciphertext<wordT>& in, const cheddar::EvkMap<wordT>& evk_map,
    const std::shared_ptr<cheddar::LinearTransform<wordT>>& transform) {
  cheddar::ConstContextPtr<wordT> cp(cheddar::ConstContextPtr<wordT>(), ctx);
  // min_ks only when the rotations form complete arithmetic progressions
  // (see CanUseLinearTransformMinKS): CHEDDAR's hoisted BSGS silently
  // corrupts transforms at deep levels (beta == 1) when key-switching with
  // chain-max keys, and its EvkMap cannot hold per-level keys; min_ks
  // selects the non-hoisted per-rotation key-switch, which (like a plain
  // hrot) is correct at any level with chain-max keys.
  transform->Evaluate(cp, out, in, evk_map,
                      /*min_ks=*/LinearTransformUsesMinKS(transform.get()));
  // Evaluate leaves the product at scale^2; rescale to the next level so the
  // result matches the ckks.rescale the lowering absorbed into this op.
  cheddar::Ciphertext<wordT> rescaled;
  ctx->Rescale(rescaled, out);
  out = std::move(rescaled);
}
template <int W, typename wordT, typename diagT>
static void RunLinearTransform(cheddar::Ciphertext<wordT>& out,
                               cheddar::Context<wordT>* ctx,
                               const cheddar::Ciphertext<wordT>& in,
                               const cheddar::EvkMap<wordT>& evk_map,
                               diagT diag[][W], std::initializer_list<int> idx,
                               int level, int bs, int gs) {
  // Heap-allocated and intentionally never destroyed: cached transforms
  // hold GPU resources, and static teardown runs after the CUDA context is
  // gone (destructing then SIGSEGVs on process exit). The key includes
  // (level, bs, gs), not just the weight array's address: the level and its
  // scale are baked into the transform at construction, and identical
  // weights CAN be evaluated at two levels (content-hashed weight
  // externalization dedupes byte-identical matrices into one array).
  static auto* cache =
      new std::map<std::tuple<const void*, int, int, int>,
                   std::shared_ptr<cheddar::LinearTransform<wordT>>>();
  auto key = std::make_tuple(static_cast<const void*>(diag), level, bs, gs);
  auto it = cache->find(key);
  if (it == cache->end()) {
    // Construct through PrepareLinearTransform so this path and the
    // split-preprocessing path share one construction site (and the MinKS
    // memoization).
    std::shared_ptr<cheddar::LinearTransform<wordT>> lt;
    PrepareLinearTransform<W>(lt, ctx, diag, idx, level, bs, gs);
    it = cache->emplace(key, std::move(lt)).first;
  }
  RunPreparedLinearTransform(out, ctx, in, evk_map, it->second);
}

#endif  // LIB_CONVERSIONS_CHEDDARTOEMITC_CHEDDARLINTRANSRUNTIME_H_
