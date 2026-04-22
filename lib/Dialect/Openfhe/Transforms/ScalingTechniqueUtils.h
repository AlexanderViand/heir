#ifndef LIB_DIALECT_OPENFHE_TRANSFORMS_SCALINGTECHNIQUEUTILS_H_
#define LIB_DIALECT_OPENFHE_TRANSFORMS_SCALINGTECHNIQUEUTILS_H_

#include "llvm/include/llvm/ADT/StringRef.h"  // from @llvm-project

namespace mlir::heir::openfhe {

inline constexpr llvm::StringLiteral kNoiseScaleDegreeAttrName =
    "openfhe.noise_scale_degree";
inline constexpr llvm::StringLiteral kScalingFactorBitsAttrName =
    "openfhe.scaling_factor_bits";
inline constexpr llvm::StringLiteral kScalingTechniqueAttrName =
    "openfhe.scaling_technique";
inline constexpr llvm::StringLiteral kNativePlaintextLevelAttrName =
    "openfhe.native_plaintext_level";

inline constexpr llvm::StringLiteral kScalingTechniqueFixedManual =
    "fixed-manual";
inline constexpr llvm::StringLiteral kScalingTechniqueFlexibleAuto =
    "flexible-auto";
inline constexpr llvm::StringLiteral kScalingTechniqueFlexibleAutoExt =
    "flexible-auto-ext";

inline bool isSupportedScalingTechnique(StringRef scalingTechnique) {
  return scalingTechnique.empty() ||
         scalingTechnique == kScalingTechniqueFixedManual ||
         scalingTechnique == kScalingTechniqueFlexibleAuto ||
         scalingTechnique == kScalingTechniqueFlexibleAutoExt;
}

inline bool usesReducedErrorPrimeSelection(StringRef scalingTechnique) {
  return scalingTechnique == kScalingTechniqueFlexibleAuto ||
         scalingTechnique == kScalingTechniqueFlexibleAutoExt;
}

inline llvm::StringRef resolveScalingTechnique(StringRef scalingTechnique) {
  return scalingTechnique.empty()
             ? llvm::StringRef(kScalingTechniqueFixedManual)
             : scalingTechnique;
}

inline bool usesExplicitPublicLevelManagement(StringRef scalingTechnique) {
  return resolveScalingTechnique(scalingTechnique) ==
         kScalingTechniqueFixedManual;
}

inline bool usesPredictiveLevelState(StringRef scalingTechnique) {
  StringRef resolved = resolveScalingTechnique(scalingTechnique);
  return resolved == kScalingTechniqueFlexibleAuto ||
         resolved == kScalingTechniqueFlexibleAutoExt;
}

}  // namespace mlir::heir::openfhe

#endif  // LIB_DIALECT_OPENFHE_TRANSFORMS_SCALINGTECHNIQUEUTILS_H_
