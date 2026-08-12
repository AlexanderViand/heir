#ifndef LIB_DIALECT_CHEDDAR_IR_CHEDDAROPS_H_
#define LIB_DIALECT_CHEDDAR_IR_CHEDDAROPS_H_

// IWYU pragma: begin_keep
#include "lib/Dialect/Cheddar/IR/CheddarDialect.h"
#include "lib/Dialect/Cheddar/IR/CheddarTypes.h"
#include "lib/Dialect/HEIRInterfaces.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"         // from @llvm-project
#include "mlir/include/mlir/Interfaces/DestinationStyleOpInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project
// IWYU pragma: end_keep

namespace mlir::heir::cheddar {

// Whether scale-snu's minimum-key-switch evaluation is valid for this BSGS
// decomposition. Both non-zero rotation sets must be complete progressions.
bool supportsMinKs(DenseI32ArrayAttr diagonalIndices, int64_t width, int64_t bs,
                   int64_t gs);

}  // namespace mlir::heir::cheddar

#define GET_OP_CLASSES
#include "lib/Dialect/Cheddar/IR/CheddarOps.h.inc"

#endif  // LIB_DIALECT_CHEDDAR_IR_CHEDDAROPS_H_
