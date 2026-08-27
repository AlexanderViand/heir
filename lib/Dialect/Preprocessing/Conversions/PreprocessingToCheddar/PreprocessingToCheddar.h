#ifndef LIB_DIALECT_PREPROCESSING_CONVERSIONS_PREPROCESSINGTOCHEDDAR_PREPROCESSINGTOCHEDDAR_H_
#define LIB_DIALECT_PREPROCESSING_CONVERSIONS_PREPROCESSINGTOCHEDDAR_PREPROCESSINGTOCHEDDAR_H_

#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"      // from @llvm-project

namespace mlir::heir::preprocessing {

#define GEN_PASS_DECL_PREPROCESSINGTOCHEDDAR
#include "lib/Dialect/Preprocessing/Conversions/PreprocessingToCheddar/PreprocessingToCheddar.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/Preprocessing/Conversions/PreprocessingToCheddar/PreprocessingToCheddar.h.inc"

}  // namespace mlir::heir::preprocessing

#endif  // LIB_DIALECT_PREPROCESSING_CONVERSIONS_PREPROCESSINGTOCHEDDAR_PREPROCESSINGTOCHEDDAR_H_
