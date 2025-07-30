#ifndef LIB_DIALECT_OPENFHE_IR_OPENFHEATTRIBUTES_H_
#define LIB_DIALECT_OPENFHE_IR_OPENFHEATTRIBUTES_H_

#include "lib/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "lib/Dialect/Openfhe/IR/OpenfheEnums.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"  // from @llvm-project

// Preserve import order
#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/Openfhe/IR/OpenfheAttributes.h.inc"

#endif  // LIB_DIALECT_OPENFHE_IR_OPENFHEATTRIBUTES_H_
