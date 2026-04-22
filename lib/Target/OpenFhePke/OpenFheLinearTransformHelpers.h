#ifndef LIB_TARGET_OPENFHEPKE_OPENFHELINEARTRANSFORMHELPERS_H_
#define LIB_TARGET_OPENFHEPKE_OPENFHELINEARTRANSFORMHELPERS_H_

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "src/core/include/math/dftransform.h"             // from @openfhe
#include "src/core/include/utils/utilities.h"              // from @openfhe
#include "src/pke/include/encoding/ckkspackedencoding.h"   // from @openfhe
#include "src/pke/include/openfhe.h"                       // from @openfhe
#include "src/pke/include/scheme/ckksrns/ckksrns-utils.h"  // from @openfhe

namespace mlir {
namespace heir {
namespace openfhe {

using OpenfheCryptoContextT = lbcrypto::CryptoContext<lbcrypto::DCRTPoly>;
using OpenfheCiphertextT = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;
using OpenfheConstCiphertextT = lbcrypto::ConstCiphertext<lbcrypto::DCRTPoly>;

inline lbcrypto::Plaintext makeOpenfheCKKSPackedPlaintextWithScalingBits(
    const OpenfheCryptoContextT& cryptoContext,
    const std::vector<double>& value, size_t noiseScaleDeg, uint32_t level,
    double scalingFactorBits, uint32_t slots = 0) {
  if (value.empty()) {
    OPENFHE_THROW("Cannot encode an empty CKKS plaintext vector.");
  }
  if (noiseScaleDeg == 0) {
    OPENFHE_THROW("CKKS plaintext encoding requires noiseScaleDeg >= 1.");
  }
  if (!std::isfinite(scalingFactorBits) || scalingFactorBits <= 0.0) {
    OPENFHE_THROW(
        "CKKS plaintext encoding requires a positive finite total "
        "scaling-factor bit count.");
  }

  auto cryptoParams =
      std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(
          cryptoContext->GetCryptoParameters());
  if (!cryptoParams) {
    OPENFHE_THROW("Expected CKKS crypto parameters for plaintext encoding.");
  }

  std::shared_ptr<lbcrypto::ILDCRTParams<lbcrypto::DCRTPoly::Integer>>
      elementParamsPtr;
  if (level != 0) {
    lbcrypto::ILDCRTParams<lbcrypto::DCRTPoly::Integer> elementParams =
        *(cryptoParams->GetElementParams());
    for (uint32_t i = 0; i < level; ++i) {
      elementParams.PopLastParam();
    }
    elementParamsPtr =
        std::make_shared<lbcrypto::ILDCRTParams<lbcrypto::DCRTPoly::Integer>>(
            elementParams);
  } else {
    elementParamsPtr = cryptoParams->GetElementParams();
  }

  std::vector<std::complex<double>> complexValue(value.size());
  std::transform(value.begin(), value.end(), complexValue.begin(),
                 [](double real) { return std::complex<double>(real, 0.0); });

  double baseScalingFactor =
      std::exp2(scalingFactorBits / static_cast<double>(noiseScaleDeg));
  lbcrypto::Plaintext plaintext =
      lbcrypto::Plaintext(std::make_shared<lbcrypto::CKKSPackedEncoding>(
          elementParamsPtr, cryptoContext->GetEncodingParams(), complexValue,
          noiseScaleDeg, level, baseScalingFactor, slots, lbcrypto::REAL));
  plaintext->Encode();
  return plaintext;
}

struct OpenfheSparseLinearTransformTerm {
  uint32_t giantStepIndex;
  uint32_t babyStepIndex;
  lbcrypto::ReadOnlyPlaintext plaintext;
};

inline void fitLinearTransformNativeVector(uint32_t ringDim,
                                           const std::vector<int64_t>& values,
                                           int64_t bigBound,
                                           lbcrypto::NativeVector* nativeVec) {
  if (nativeVec == nullptr) {
    OPENFHE_THROW("The passed native vector is empty.");
  }
  lbcrypto::NativeInteger halfBound(bigBound >> 1);
  lbcrypto::NativeInteger modulus(nativeVec->GetModulus());
  lbcrypto::NativeInteger diff = bigBound - modulus;
  uint32_t gap = ringDim / values.size();
  for (uint32_t i = 0; i < values.size(); ++i) {
    lbcrypto::NativeInteger n(values[i]);
    if (n > halfBound) {
      (*nativeVec)[gap * i] = n.ModSub(diff, modulus);
    } else {
      (*nativeVec)[gap * i] = n.Mod(modulus);
    }
  }
}

inline lbcrypto::Plaintext makeOpenfheLinearTransformAuxPlaintext(
    const OpenfheCryptoContextT& cryptoContext,
    const std::shared_ptr<lbcrypto::ILDCRTParams<lbcrypto::DCRTPoly::Integer>>&
        params,
    const std::vector<std::complex<double>>& value, size_t noiseScaleDeg,
    uint32_t level, uint32_t slots) {
  auto cryptoParams =
      std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(
          cryptoContext->GetCryptoParameters());
  double scalingFactor = cryptoParams->GetScalingFactorReal(level);

  lbcrypto::Plaintext plaintext =
      lbcrypto::Plaintext(std::make_shared<lbcrypto::CKKSPackedEncoding>(
          params, cryptoContext->GetEncodingParams(), value, noiseScaleDeg,
          level, scalingFactor, slots, lbcrypto::COMPLEX));

  lbcrypto::DCRTPoly& plainElement =
      plaintext->GetElement<lbcrypto::DCRTPoly>();
  uint32_t ringDim = cryptoContext->GetRingDimension();

  std::vector<std::complex<double>> inverse = value;
  inverse.resize(slots);
  lbcrypto::DiscreteFourierTransform::FFTSpecialInv(inverse, ringDim * 2);

  double powP = scalingFactor;
  constexpr int32_t kMaxBitsInWord = 61;
  int32_t logc = std::numeric_limits<int32_t>::min();
  for (uint32_t i = 0; i < slots; ++i) {
    inverse[i] *= powP;
    if (inverse[i].real() != 0.0) {
      int32_t logci = static_cast<int32_t>(
          std::ceil(std::log2(std::abs(inverse[i].real()))));
      logc = std::max(logc, logci);
    }
    if (inverse[i].imag() != 0.0) {
      int32_t logci = static_cast<int32_t>(
          std::ceil(std::log2(std::abs(inverse[i].imag()))));
      logc = std::max(logc, logci);
    }
  }
  logc = (logc == std::numeric_limits<int32_t>::min()) ? 0 : logc;
  if (logc < 0) {
    OPENFHE_THROW("Scaling factor too small");
  }

  int32_t logValid = (logc <= kMaxBitsInWord) ? logc : kMaxBitsInWord;
  int32_t logApprox = logc - logValid;
  double approxFactor = std::pow(2.0, logApprox);

  std::vector<int64_t> temp(2 * slots);
  for (size_t i = 0; i < slots; ++i) {
    double realPart = inverse[i].real() / approxFactor;
    double imagPart = inverse[i].imag() / approxFactor;
    if (lbcrypto::is64BitOverflow(realPart) ||
        lbcrypto::is64BitOverflow(imagPart)) {
      OPENFHE_THROW(
          "Overflow in OpenFHE linear_transform aux plaintext encoding");
    }

    int64_t realRounded = std::llround(realPart);
    int64_t imagRounded = std::llround(imagPart);
    temp[i] = (realRounded < 0) ? lbcrypto::Max64BitValue() + realRounded
                                : realRounded;
    temp[i + slots] = (imagRounded < 0)
                          ? lbcrypto::Max64BitValue() + imagRounded
                          : imagRounded;
  }

  auto bigParams = plainElement.GetParams();
  const auto& nativeParams = bigParams->GetParams();
  for (size_t i = 0; i < nativeParams.size(); ++i) {
    lbcrypto::NativeVector nativeVec(ringDim, nativeParams[i]->GetModulus());
    fitLinearTransformNativeVector(ringDim, temp, lbcrypto::Max64BitValue(),
                                   &nativeVec);
    lbcrypto::NativePoly element = plainElement.GetElementAtIndex(i);
    element.SetValues(std::move(nativeVec), Format::COEFFICIENT);
    plainElement.SetElementAtIndex(i, std::move(element));
  }

  uint32_t numTowers = nativeParams.size();
  std::vector<lbcrypto::DCRTPoly::Integer> moduli(numTowers);
  for (uint32_t i = 0; i < numTowers; ++i) {
    moduli[i] = nativeParams[i]->GetModulus();
  }

  lbcrypto::DCRTPoly::Integer intPowP{
      static_cast<uint64_t>(std::llround(powP))};
  std::vector<lbcrypto::DCRTPoly::Integer> crtPowP(numTowers, intPowP);
  auto currentPowP = crtPowP;
  for (size_t i = 2; i < noiseScaleDeg; ++i) {
    currentPowP =
        lbcrypto::CKKSPackedEncoding::CRTMult(currentPowP, crtPowP, moduli);
  }
  if (noiseScaleDeg > 1) {
    plainElement = plainElement.Times(currentPowP);
  }

  if (logApprox > 0) {
    int32_t logStep = (logApprox <= lbcrypto::MAX_LOG_STEP)
                          ? logApprox
                          : lbcrypto::MAX_LOG_STEP;
    auto intStep =
        lbcrypto::DCRTPoly::Integer(static_cast<uint64_t>(1) << logStep);
    std::vector<lbcrypto::DCRTPoly::Integer> crtApprox(numTowers, intStep);
    logApprox -= logStep;
    while (logApprox > 0) {
      logStep = (logApprox <= lbcrypto::MAX_LOG_STEP) ? logApprox
                                                      : lbcrypto::MAX_LOG_STEP;
      intStep =
          lbcrypto::DCRTPoly::Integer(static_cast<uint64_t>(1) << logStep);
      std::vector<lbcrypto::DCRTPoly::Integer> crtScale(numTowers, intStep);
      crtApprox =
          lbcrypto::CKKSPackedEncoding::CRTMult(crtApprox, crtScale, moduli);
      logApprox -= logStep;
    }
    plainElement = plainElement.Times(crtApprox);
  }

  plaintext->SetFormat(Format::EVALUATION);
  plaintext->SetScalingFactor(
      std::pow(plaintext->GetScalingFactor(), noiseScaleDeg));
  return plaintext;
}

inline std::vector<OpenfheSparseLinearTransformTerm>
precomputeOpenfheSparseLinearTransform(
    const OpenfheCryptoContextT& cryptoContext,
    const std::vector<double>& diagonalsFlat,
    const std::vector<int32_t>& diagonalIndices, uint32_t slots,
    uint32_t babyStep, uint32_t plaintextLevel, double diagonalScale = 1.0) {
  if (slots == 0) {
    OPENFHE_THROW(
        "OpenFHE linear_transform precompute requires a non-zero slot count");
  }
  if (diagonalIndices.empty()) {
    OPENFHE_THROW(
        "OpenFHE linear_transform precompute requires at least one diagonal");
  }
  if (diagonalsFlat.size() != diagonalIndices.size() * slots) {
    OPENFHE_THROW(
        "OpenFHE linear_transform diagonal tensor size does not "
        "match diagonal_indices and slot count");
  }
  auto cryptoParams =
      std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(
          cryptoContext->GetCryptoParameters());
  auto elementParams = *cryptoParams->GetElementParams();
  uint32_t compositeDegree = cryptoParams->GetCompositeDegree();
  uint32_t towersToDrop = 0;
  if (plaintextLevel != 0) {
    towersToDrop =
        elementParams.GetParams().size() - plaintextLevel - compositeDegree;
    for (uint32_t i = 0; i < towersToDrop; ++i) {
      elementParams.PopLastParam();
    }
  }

  const auto& paramsQ = elementParams.GetParams();
  const auto& paramsP = cryptoParams->GetParamsP()->GetParams();
  std::vector<lbcrypto::NativeInteger> moduli;
  std::vector<lbcrypto::NativeInteger> roots;
  moduli.reserve(paramsQ.size() + paramsP.size());
  roots.reserve(paramsQ.size() + paramsP.size());
  for (const auto& elem : paramsQ) {
    moduli.emplace_back(elem->GetModulus());
    roots.emplace_back(elem->GetRootOfUnity());
  }
  for (const auto& elem : paramsP) {
    moduli.emplace_back(elem->GetModulus());
    roots.emplace_back(elem->GetRootOfUnity());
  }

  auto extendedParams =
      std::make_shared<lbcrypto::ILDCRTParams<lbcrypto::DCRTPoly::Integer>>(
          cryptoContext->GetCyclotomicOrder(), moduli, roots);

  std::vector<OpenfheSparseLinearTransformTerm> result;
  result.reserve(diagonalIndices.size());
  for (size_t row = 0; row < diagonalIndices.size(); ++row) {
    int64_t normalized =
        ((static_cast<int64_t>(diagonalIndices[row]) % slots) + slots) % slots;
    uint32_t giantStepIndex = static_cast<uint32_t>(normalized / babyStep);
    uint32_t babyStepIndex = static_cast<uint32_t>(normalized % babyStep);
    int32_t offset = -static_cast<int32_t>(babyStep * giantStepIndex);

    std::vector<std::complex<double>> diag(slots);
    for (uint32_t i = 0; i < slots; ++i) {
      diag[i] = std::complex<double>(
          diagonalsFlat[row * static_cast<size_t>(slots) + i] * diagonalScale,
          0.0);
    }
    result.push_back(OpenfheSparseLinearTransformTerm{
        giantStepIndex, babyStepIndex,
        makeOpenfheLinearTransformAuxPlaintext(cryptoContext, extendedParams,
                                               lbcrypto::Rotate(diag, offset),
                                               1, towersToDrop, diag.size())});
  }
  std::sort(result.begin(), result.end(),
            [](const OpenfheSparseLinearTransformTerm& lhs,
               const OpenfheSparseLinearTransformTerm& rhs) {
              return std::pair(lhs.giantStepIndex, lhs.babyStepIndex) <
                     std::pair(rhs.giantStepIndex, rhs.babyStepIndex);
            });
  return result;
}

inline OpenfheCiphertextT evalOpenfheLinearTransformWithPrecompute(
    const OpenfheCryptoContextT& cryptoContext, OpenfheConstCiphertextT input,
    const std::vector<OpenfheSparseLinearTransformTerm>& precomputed,
    uint32_t babyStep) {
  if (precomputed.empty()) {
    OPENFHE_THROW(
        "OpenFHE linear_transform requires at least one precomputed diagonal");
  }
  auto evalMultExt = [](OpenfheConstCiphertextT ciphertext,
                        lbcrypto::ConstPlaintext plaintext) {
    auto plain = plaintext->GetElement<lbcrypto::DCRTPoly>();
    plain.SetFormat(Format::EVALUATION);
    auto result = ciphertext->Clone();
    const auto& plainParams = plain.GetParams()->GetParams();
    for (auto& elem : result->GetElements()) {
      const auto& elemParams = elem.GetParams()->GetParams();
      if (elemParams.size() != plainParams.size()) {
        OPENFHE_THROW(
            "OpenFHE linear_transform ciphertext/plaintext tower "
            "count mismatch: ciphertext has " +
            std::to_string(elemParams.size()) +
            " towers but aux plaintext has " +
            std::to_string(plainParams.size()) + " towers (ciphertext level=" +
            std::to_string(ciphertext->GetLevel()) +
            ", ciphertext noiseScaleDeg=" +
            std::to_string(ciphertext->GetNoiseScaleDeg()) +
            ", plaintext level=" + std::to_string(plaintext->GetLevel()) +
            ", plaintext noiseScaleDeg=" +
            std::to_string(plaintext->GetNoiseScaleDeg()) + ")");
      }
      elem *= plain;
    }
    result->SetNoiseScaleDeg(result->GetNoiseScaleDeg() +
                             plaintext->GetNoiseScaleDeg());
    result->SetScalingFactor(result->GetScalingFactor() *
                             plaintext->GetScalingFactor());
    return result;
  };
  auto evalAddExtInPlace = [](OpenfheCiphertextT& lhs,
                              OpenfheConstCiphertextT rhs) {
    auto& lhsElements = lhs->GetElements();
    const auto& rhsElements = rhs->GetElements();
    for (size_t i = 0; i < lhsElements.size(); ++i) {
      lhsElements[i] += rhsElements[i];
    }
  };
  uint32_t cyclotomicOrder = cryptoContext->GetCyclotomicOrder();
  uint32_t ringDim = cryptoContext->GetRingDimension();

  auto digits = cryptoContext->EvalFastRotationPrecompute(input);
  std::vector<bool> usedBabySteps(babyStep, false);
  bool needsUnrotatedExt = false;
  for (const auto& term : precomputed) {
    if (term.babyStepIndex == 0) {
      needsUnrotatedExt = true;
    } else {
      usedBabySteps[term.babyStepIndex] = true;
    }
  }
  OpenfheCiphertextT unrotatedExt;
  if (needsUnrotatedExt) {
    unrotatedExt = cryptoContext->KeySwitchExt(input, true);
  }
  std::vector<OpenfheCiphertextT> fastRotations(babyStep);
  for (uint32_t j = 1; j < babyStep; ++j) {
    if (!usedBabySteps[j]) {
      continue;
    }
    fastRotations[j] =
        cryptoContext->EvalFastRotationExt(input, j, digits, true);
  }

  OpenfheCiphertextT result;
  lbcrypto::DCRTPoly first;
  bool haveResult = false;
  bool haveFirst = false;
  size_t index = 0;
  while (index < precomputed.size()) {
    uint32_t giantStepIndex = precomputed[index].giantStepIndex;
    OpenfheCiphertextT inner;
    bool haveInner = false;
    while (index < precomputed.size() &&
           precomputed[index].giantStepIndex == giantStepIndex) {
      const auto& term = precomputed[index];
      OpenfheConstCiphertextT source = term.babyStepIndex == 0
                                           ? unrotatedExt
                                           : fastRotations[term.babyStepIndex];
      auto termProduct = evalMultExt(source, term.plaintext);
      if (!haveInner) {
        inner = std::move(termProduct);
        haveInner = true;
      } else {
        evalAddExtInPlace(inner, termProduct);
      }
      ++index;
    }
    if (!haveInner) {
      continue;
    }

    if (giantStepIndex == 0) {
      auto firstContribution = cryptoContext->KeySwitchDownFirstElement(inner);
      if (!haveFirst) {
        first = firstContribution;
        haveFirst = true;
      } else {
        first += firstContribution;
      }
      auto elements = inner->GetElements();
      elements[0].SetValuesToZero();
      inner->SetElements(std::move(elements));
      if (!haveResult) {
        result = std::move(inner);
        haveResult = true;
      } else {
        evalAddExtInPlace(result, inner);
      }
      continue;
    }

    inner = cryptoContext->KeySwitchDown(inner);
    uint32_t autoIndex = lbcrypto::FindAutomorphismIndex2nComplex(
        babyStep * giantStepIndex, cyclotomicOrder);
    std::vector<uint32_t> map(ringDim);
    lbcrypto::PrecomputeAutoMap(ringDim, autoIndex, &map);
    auto firstContribution =
        inner->GetElements()[0].AutomorphismTransform(autoIndex, map);
    if (!haveFirst) {
      first = firstContribution;
      haveFirst = true;
    } else {
      first += firstContribution;
    }

    auto innerDigits = cryptoContext->EvalFastRotationPrecompute(inner);
    auto rotated = cryptoContext->EvalFastRotationExt(
        inner, babyStep * giantStepIndex, innerDigits, false);
    if (!haveResult) {
      result = std::move(rotated);
      haveResult = true;
    } else {
      evalAddExtInPlace(result, rotated);
    }
  }

  if (!haveResult || !haveFirst) {
    throw std::runtime_error(
        "OpenFHE linear_transform failed to construct a result");
  }
  result = cryptoContext->KeySwitchDown(result);
  result->GetElements()[0] += first;
  // The EXT-basis BSGS loop accumulates noiseScaleDeg through intermediate
  // EvalMultExt calls. Reset to the correct value: the linear_transform
  // consumes exactly one plaintext scale factor, so nsd increments by 1.
  result->SetNoiseScaleDeg(input->GetNoiseScaleDeg() + 1);
  return result;
}

inline OpenfheCiphertextT evalOpenfheSparseLinearTransform(
    const OpenfheCryptoContextT& cryptoContext, OpenfheConstCiphertextT input,
    const std::vector<double>& diagonalsFlat,
    const std::vector<int32_t>& diagonalIndices, uint32_t slots,
    uint32_t babyStep, uint32_t plaintextLevel) {
  // Derive the actual plaintext level from the input ciphertext's runtime
  // state. This is critical for FLEXIBLE* modes where auto-rescaling shifts
  // the ciphertext's level beyond what compile-time analysis predicts.
  auto cryptoParams =
      std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(
          cryptoContext->GetCryptoParameters());
  uint32_t sizeQ = cryptoParams->GetElementParams()->GetParams().size();
  uint32_t compositeDegree = cryptoParams->GetCompositeDegree();
  uint32_t ctLevel = input->GetLevel();
  uint32_t runtimePlaintextLevel = sizeQ - ctLevel - compositeDegree;
  auto precomputed = precomputeOpenfheSparseLinearTransform(
      cryptoContext, diagonalsFlat, diagonalIndices, slots, babyStep,
      runtimePlaintextLevel);
  return evalOpenfheLinearTransformWithPrecompute(cryptoContext, input,
                                                  precomputed, babyStep);
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_OPENFHEPKE_OPENFHELINEARTRANSFORMHELPERS_H_
