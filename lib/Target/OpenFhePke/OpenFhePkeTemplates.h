#ifndef LIB_TARGET_OPENFHEPKE_OPENFHEPKETEMPLATES_H_
#define LIB_TARGET_OPENFHEPKE_OPENFHEPKETEMPLATES_H_

#include <string_view>

namespace mlir {
namespace heir {
namespace openfhe {

constexpr std::string_view kSourceRelativeOpenfheImport = R"cpp(
#include <complex>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "src/pke/include/openfhe.h"  // from @openfhe
)cpp";
constexpr std::string_view kInstallationRelativeOpenfheImport = R"cpp(
#include <complex>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "openfhe/pke/openfhe.h"  // from @openfhe
)cpp";
constexpr std::string_view kEmbeddedOpenfheImport = R"cpp(
#include <complex>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "openfhe.h"
)cpp";

// clang-format off
constexpr std::string_view kModulePreludeTemplate = R"cpp(
using namespace lbcrypto;
using CiphertextT = Ciphertext<DCRTPoly>;
using ConstCiphertextT = ConstCiphertext<DCRTPoly>;
using CCParamsT = CCParams<CryptoContext{0}RNS>;
using CryptoContextT = CryptoContext<DCRTPoly>;
using EvalKeyT = EvalKey<DCRTPoly>;
using PlaintextT = Plaintext;
using ReadOnlyPlaintextT = ReadOnlyPlaintext;
using PrivateKeyT = PrivateKey<DCRTPoly>;
using PublicKeyT = PublicKey<DCRTPoly>;
)cpp";
// clang-format on

// clang-format off
constexpr std::string_view kModuleHelperPrelude = R"cpp(
inline std::string heir_cache_key(const char* base, CryptoContextT cc) {
  return std::string(base) + "@" +
         std::to_string(reinterpret_cast<std::uintptr_t>(cc.get()));
}

struct HeirLinearTransform {
  bool initialized = false;
  uint32_t baby_step = 0;
  std::vector<int32_t> diagonal_indices;
  std::vector<ReadOnlyPlaintextT> plaintexts;
};

inline std::map<std::string, HeirLinearTransform>&
heir_linear_transform_cache() {
  static std::map<std::string, HeirLinearTransform> cache;
  return cache;
}

inline std::map<std::string, PlaintextT>& heir_ckks_plaintext_cache() {
  static std::map<std::string, PlaintextT> cache;
  return cache;
}

inline int64_t heir_wrap_rotation(int64_t rot, int64_t slots) {
  int64_t wrapped = rot % slots;
  return wrapped < 0 ? wrapped + slots : wrapped;
}

inline uint32_t heir_find_best_bsgs_ratio(
    const std::vector<int32_t>& diagonal_indices, int64_t slots,
    int64_t log_max_ratio) {
  int64_t max_ratio = 1LL << log_max_ratio;
  for (int64_t n1 = 1; n1 < slots; n1 <<= 1) {
    std::map<int64_t, bool> rot_n1_set;
    std::map<int64_t, bool> rot_n2_set;
    for (auto rot : diagonal_indices) {
      int64_t r = heir_wrap_rotation(rot, slots);
      rot_n1_set[((r / n1) * n1) & (slots - 1)] = true;
      rot_n2_set[r & (n1 - 1)] = true;
    }
    int64_t nb_n1 = static_cast<int64_t>(rot_n1_set.size()) - 1;
    int64_t nb_n2 = static_cast<int64_t>(rot_n2_set.size()) - 1;
    if (nb_n1 > 0) {
      if (nb_n2 == max_ratio * nb_n1) return static_cast<uint32_t>(n1);
      if (nb_n2 > max_ratio * nb_n1) return static_cast<uint32_t>(n1 / 2);
    }
  }
  return 1;
}

template <typename FloatT>
inline std::vector<std::vector<std::complex<double>>>
heir_make_sparse_diagonals(const std::vector<FloatT>& flat_diagonals,
                           int64_t diagonal_count, int64_t slot_count) {
  std::vector<std::vector<std::complex<double>>> diagonals(
      diagonal_count, std::vector<std::complex<double>>(slot_count));
  for (int64_t diagonal = 0; diagonal < diagonal_count; ++diagonal) {
    for (int64_t slot = 0; slot < slot_count; ++slot) {
      diagonals[diagonal][slot] = std::complex<double>(
          flat_diagonals[diagonal * slot_count + slot], 0.0);
    }
  }
  return diagonals;
}

template <typename FloatT>
inline HeirLinearTransform heir_precompute_linear_transform(
    CryptoContextT cc, const std::vector<FloatT>& flat_diagonals,
    const std::vector<int32_t>& diagonal_indices, int64_t log_bsgs_ratio,
    uint32_t level) {
  auto scheme = std::dynamic_pointer_cast<SchemeCKKSRNS>(cc->GetScheme());
  if (!scheme) {
    OPENFHE_THROW("HEIR expected a CKKS SchemeCKKSRNS scheme");
  }
  if (diagonal_indices.empty()) {
    OPENFHE_THROW("HEIR linear transform precompute requires diagonals");
  }

  int64_t diagonal_count = static_cast<int64_t>(diagonal_indices.size());
  int64_t slot_count = static_cast<int64_t>(flat_diagonals.size()) /
                       diagonal_count;
  auto sparse_diagonals =
      heir_make_sparse_diagonals(flat_diagonals, diagonal_count, slot_count);

  HeirLinearTransform result;
  result.initialized = true;
  result.diagonal_indices = diagonal_indices;
  result.baby_step = (log_bsgs_ratio < 0)
                         ? static_cast<uint32_t>(slot_count)
                         : heir_find_best_bsgs_ratio(diagonal_indices,
                                                     slot_count,
                                                     log_bsgs_ratio);
  result.plaintexts = scheme->EvalLinearTransformPrecomputeSparse(
      *cc.get(), sparse_diagonals, diagonal_indices, result.baby_step, 1.0,
      level);
  return result;
}

inline CiphertextT heir_eval_linear_transform(CryptoContextT cc,
                                              ConstCiphertextT ciphertext,
                                              const HeirLinearTransform& lt) {
  auto scheme = std::dynamic_pointer_cast<SchemeCKKSRNS>(cc->GetScheme());
  if (!scheme) {
    OPENFHE_THROW("HEIR expected a CKKS SchemeCKKSRNS scheme");
  }
  return scheme->EvalLinearTransformSparse(lt.plaintexts, ciphertext,
                                           lt.diagonal_indices, lt.baby_step);
}
)cpp";
// clang-format on

// clang-format off
constexpr std::string_view kWeightsPreludeTemplate = R"cpp(
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "include/cereal/archives/portable_binary.hpp"  // from @cereal
#include "include/cereal/cereal.hpp"  // from @cereal

struct Weights {
  std::map<std::string, std::vector<float>> floats;
  std::map<std::string, std::vector<double>> doubles;
  std::map<std::string, std::vector<int64_t>> int64_ts;
  std::map<std::string, std::vector<int32_t>> int32_ts;
  std::map<std::string, std::vector<int16_t>> int16_ts;
  std::map<std::string, std::vector<int8_t>> int8_ts;

  template <class Archive>
  void serialize(Archive &archive) {
    archive(CEREAL_NVP(floats), CEREAL_NVP(doubles), CEREAL_NVP(int64_ts),
            CEREAL_NVP(int32_ts), CEREAL_NVP(int16_ts), CEREAL_NVP(int8_ts));
  }
};

Weights GetWeightModule(const std::string& filename) {
  Weights obj;
  std::ifstream file(filename, std::ios::in | std::ios::binary);
  cereal::PortableBinaryInputArchive archive(file);
  archive(obj);
  file.close();
  return obj;
}
)cpp";
// clang-format on

// clang-format off
constexpr std::string_view kPybindImports = R"cpp(
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
)cpp";
// clang-format on

// clang-format off
constexpr std::string_view kPybindCommon = R"cpp(
using namespace lbcrypto;
namespace py = pybind11;

CiphertextT encrypt_ckks(CryptoContextT cc, const std::vector<double>& values,
                         PublicKeyT pk) {
    auto pt = cc->MakeCKKSPackedPlaintext(values);
    return cc->Encrypt(pk, pt);
}

std::vector<double> decrypt_ckks(CryptoContextT cc, CiphertextT ct,
                                 PrivateKeyT sk, size_t length) {
    PlaintextT pt;
    cc->Decrypt(sk, ct, &pt);
    pt->SetLength(length);
    return pt->GetRealPackedValue();
}

// Minimal bindings required for generated functions to run.
// Cf. https://pybind11.readthedocs.io/en/stable/advanced/classes.html#module-local-class-bindings
// which is a temporary workaround to allow us to have multiple compilations in
// the same python program. Better would be to cache the pybind11 module across
// calls.
void bind_common(py::module &m)
{
    py::class_<PublicKeyImpl<DCRTPoly>, std::shared_ptr<PublicKeyImpl<DCRTPoly>>>(m, "PublicKey", py::module_local())
        .def(py::init<>());
    py::class_<PrivateKeyImpl<DCRTPoly>, std::shared_ptr<PrivateKeyImpl<DCRTPoly>>>(m, "PrivateKey", py::module_local())
        .def(py::init<>());
    py::class_<KeyPair<DCRTPoly>>(m, "KeyPair", py::module_local())
        .def_readwrite("publicKey", &KeyPair<DCRTPoly>::publicKey)
        .def_readwrite("secretKey", &KeyPair<DCRTPoly>::secretKey);
    py::class_<CiphertextImpl<DCRTPoly>, std::shared_ptr<CiphertextImpl<DCRTPoly>>>(m, "Ciphertext", py::module_local())
        .def(py::init<>());
    py::class_<CryptoContextImpl<DCRTPoly>, std::shared_ptr<CryptoContextImpl<DCRTPoly>>>(m, "CryptoContext", py::module_local())
        .def(py::init<>())
        .def("KeyGen", &CryptoContextImpl<DCRTPoly>::KeyGen);
}
)cpp";
// clang-format on

// clang-format off
constexpr std::string_view kPybindModuleTemplate = R"cpp(
PYBIND11_MODULE({0}, m) {{
  bind_common(m);
  m.def("encrypt_ckks", &encrypt_ckks,
        py::call_guard<py::gil_scoped_release>());
  m.def("decrypt_ckks", &decrypt_ckks,
        py::call_guard<py::gil_scoped_release>());
)cpp";
// clang-format on

// Emit a pybind11 binding that releases the GIL for the duration of the C++
// function call.  This enables multi-threaded C++ code (e.g. OpenMP parallel
// regions inside OpenFHE) to run concurrently with the Python interpreter.
// The `py::call_guard<py::gil_scoped_release>()` helper ensures the GIL is
// relinquished on entry and re-acquired on exit.
constexpr std::string_view kPybindFunctionTemplate =
    "m.def(\"{0}\", &{0}, py::call_guard<py::gil_scoped_release>());";

// clang-format off
constexpr std::string_view KdebugHeaderImports = R"cpp(
#include <map>
#include <string>
)cpp";
// clang-format on

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_OPENFHEPKE_OPENFHEPKETEMPLATES_H_
