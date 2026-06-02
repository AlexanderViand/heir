#include "lib/Target/OpenFhePke/OpenFhePkePybindEmitter.h"

#include <string>

#include "lib/Target/OpenFhePke/OpenFhePkeTemplates.h"
#include "lib/Utils/TargetUtils.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"           // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"   // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project

namespace mlir {
namespace heir {
namespace openfhe {

LogicalResult translateToOpenFhePkePybind(Operation* op, llvm::raw_ostream& os,
                                          const std::string& headerInclude,
                                          const std::string& pythonModuleName) {
  OpenFhePkePybindEmitter emitter(os, headerInclude, pythonModuleName);
  return emitter.translate(*op);
}

LogicalResult OpenFhePkePybindEmitter::translate(Operation& op) {
  LogicalResult status =
      llvm::TypeSwitch<Operation&, LogicalResult>(op)
          .Case<ModuleOp>([&](auto op) { return printOperation(op); })
          .Case<func::FuncOp>([&](auto op) { return printOperation(op); })
          .Default([&](Operation&) {
            return op.emitOpError("unable to find printer for op");
          });

  if (failed(status)) {
    op.emitOpError(llvm::formatv("Failed to translate op {0}", op.getName()));
    return failure();
  }
  return success();
}

LogicalResult OpenFhePkePybindEmitter::printOperation(ModuleOp moduleOp) {
  os << kPybindImports << "\n";
  os << "#include \"" << headerInclude_ << "\"\n";
  os << kPybindCommon << "\n";

  os << llvm::formatv(kPybindModuleTemplate.data(), pythonModuleName_) << "\n";
  os.indent();

  for (Operation& op : moduleOp) {
    if (failed(translate(op))) {
      return failure();
    }
  }

  os.unindent();
  os << "}\n";
  return success();
}

LogicalResult OpenFhePkePybindEmitter::printOperation(func::FuncOp funcOp) {
  StringRef name = canonicalizeDebugPort(funcOp.getName());

  // A function with multiple results returns a generated aggregate named
  // "<name>Struct" with fields arg0..arg{N-1} (see OpenFhePkeEmitter). Bind it
  // as a py::class_ with read/write fields so Python can construct and unpack
  // it -- e.g. to encode plaintexts once via a "__preprocessing" function and
  // pass them into a split-out "__preprocessed" compute function.
  unsigned numResults = funcOp.getNumResults();
  if (numResults > 1) {
    os << llvm::formatv("py::class_<{0}Struct>(m, \"{0}Struct\")", name);
    for (unsigned i = 0; i < numResults; ++i) {
      os << llvm::formatv(".def_readwrite(\"arg{0}\", &{1}Struct::arg{0})", i,
                          name);
    }
    os << ";\n";
  }

  os << llvm::formatv(kPybindFunctionTemplate.data(), name) << "\n";
  return success();
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
