#ifndef LIB_TRANSFORMS_ADDMEMREFALIGNMENT_ADDMEMREFALIGNMENT_H_
#define LIB_TRANSFORMS_ADDMEMREFALIGNMENT_ADDMEMREFALIGNMENT_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DECL_ADDMEMREFALIGNMENTPASS
#include "lib/Transforms/AddMemrefAlignment/AddMemrefAlignment.h.inc"

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_ADDMEMREFALIGNMENT_ADDMEMREFALIGNMENT_H_
