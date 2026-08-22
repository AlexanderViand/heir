#ifndef LIB_DIALECT_CHEDDAR_IR_CHEDDARTYPES_H_
#define LIB_DIALECT_CHEDDAR_IR_CHEDDARTYPES_H_

// IWYU pragma: begin_keep
#include "lib/Dialect/Cheddar/IR/CheddarDialect.h"
#include "lib/Dialect/HEIRInterfaces.h"
#include "mlir/include/mlir/IR/OpImplementation.h"  // from @llvm-project
// IWYU pragma: end_keep

#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/Cheddar/IR/CheddarTypes.h.inc"

namespace mlir {
namespace heir {
namespace cheddar {

// Argument attribute naming the kind of a lowered support argument (a context,
// encoder, key, ...). Its value is the support type's mnemonic, so later
// stages read the role off the argument instead of recovering it from the
// type, or after EmitC conversion from a C++ type name.
constexpr ::llvm::StringLiteral kSupportArgAttrName = "cheddar.support";

// The support kind of `type`, or empty when it is not a support type.
::llvm::StringRef getSupportKind(::mlir::Type type);

// The support kind of the resource directory a preprocessing function loads
// from (see preprocessing-thread-resource-dir).
constexpr ::llvm::StringLiteral kResourceDirSupportKind = "resource_dir";

// Argument attributes on the facades cheddar-build-entry-interface generates:
// the entry argument a data argument is fed from, and the data arguments that
// hold preprocessed values. A facade's first argument is the context its
// caller owns.
constexpr ::llvm::StringLiteral kEntryInputArgAttrName = "cheddar.entry_input";
constexpr ::llvm::StringLiteral kPreparedArgAttrName = "cheddar.prepared";

}  // namespace cheddar
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_CHEDDAR_IR_CHEDDARTYPES_H_
