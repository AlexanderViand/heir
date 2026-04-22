#ifndef LIB_TRANSFORMS_POPULATESCALE_RESOLVEOPENFHECKKSMETADATA_H_
#define LIB_TRANSFORMS_POPULATESCALE_RESOLVEOPENFHECKKSMETADATA_H_

#include "llvm/include/llvm/ADT/StringRef.h"          // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir::heir::openfhe {

// Resolves OpenFHE-specific CKKS metadata on already managed CKKS/Orion IR.
// This may:
// - enlarge ckks.schemeParam when the chosen OpenFHE scaling technique needs a
//   longer chain than generic CKKS management predicted,
// - annotate opaque native linear transforms with their required plaintext
//   level,
// - and annotate add-plain encodes with the metadata the OpenFHE lowering must
//   preserve.
LogicalResult resolveOpenfheCKKSMetadata(ModuleOp module,
                                         llvm::StringRef scalingTechnique);

}  // namespace mlir::heir::openfhe

#endif  // LIB_TRANSFORMS_POPULATESCALE_RESOLVEOPENFHECKKSMETADATA_H_
