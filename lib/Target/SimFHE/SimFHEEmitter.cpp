#include "lib/Target/SimFHE/SimFHEEmitter.h"

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/ModArith/IR/ModArithDialect.h"
#include "lib/Dialect/Polynomial/IR/PolynomialDialect.h"
#include "lib/Dialect/RNS/IR/RNSDialect.h"
#include "lib/Dialect/RNS/IR/RNSTypeInterfaces.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"          // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"  // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"      // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace simfhe {

void registerToSimFHETranslation() {
  TranslateFromMLIRRegistration reg(
      "emit-simfhe", "translate ckks dialect to SimFHE python code",
      [](Operation *op, llvm::raw_ostream &output) {
        return translateToSimFHE(op, output);
      },
      [](DialectRegistry &registry) {
        registry.insert<func::FuncDialect, ckks::CKKSDialect, lwe::LWEDialect,
                        polynomial::PolynomialDialect,
                        mod_arith::ModArithDialect, rns::RNSDialect>();
        rns::registerExternalRNSTypeInterfaces(registry);
      });
}

LogicalResult translateToSimFHE(Operation *op, llvm::raw_ostream &os) {
  SelectVariableNames variableNames(op);
  SimFHEEmitter emitter(os, &variableNames);
  return emitter.translate(*op);
}

SimFHEEmitter::SimFHEEmitter(llvm::raw_ostream &os,
                             SelectVariableNames *variableNames)
    : os(os), variableNames(variableNames) {}

LogicalResult SimFHEEmitter::translate(Operation &op) {
  LogicalResult status =
      llvm::TypeSwitch<Operation &, LogicalResult>(op)
          .Case<ModuleOp, func::FuncOp, func::ReturnOp, ckks::AddOp,
                ckks::AddPlainOp, ckks::SubOp, ckks::SubPlainOp, ckks::MulOp,
                ckks::MulPlainOp, ckks::NegateOp, ckks::RotateOp,
                ckks::RelinearizeOp, ckks::RescaleOp, ckks::LevelReduceOp,
                ckks::BootstrapOp>(
              [&](auto innerOp) { return printOperation(innerOp); })
          .Default([&](Operation &) {
            return op.emitOpError("unable to find printer for op");
          });
  if (failed(status)) {
    return op.emitOpError(
        llvm::formatv("Failed to translate op {0}", op.getName()));
  }
  return success();
}

LogicalResult SimFHEEmitter::printOperation(ModuleOp moduleOp) {
  os << kModulePrelude << "\n";
  for (Operation &op : moduleOp) {
    if (failed(translate(op))) {
      return failure();
    }
  }
  return success();
}

LogicalResult SimFHEEmitter::printOperation(func::FuncOp funcOp) {
  os << "def " << funcOp.getName() << "(";
  os << heir::commaSeparatedValues(funcOp.getArguments(),
                                   [&](Value value) { return getName(value); });
  os << "):\n";
  os.indent();
  os << "stats = PerfCounter()\n";
  for (Block &block : funcOp.getBlocks()) {
    for (Operation &op : block.getOperations()) {
      if (failed(translate(op))) {
        return failure();
      }
    }
  }
  os << "return stats\n";
  os.unindent();
  os << "\n";
  return success();
}

LogicalResult SimFHEEmitter::printOperation(func::ReturnOp op) {
  // return is handled via accumulating stats; nothing to do
  return success();
}

#define UNARY_EMIT(OPTYPE, FUNCNAME)                               \
  LogicalResult SimFHEEmitter::printOperation(OPTYPE op) {         \
    os << "stats += evaluator." FUNCNAME "("                       \
       << getName(op.getOperands().front()) << ", arch_params)\n"; \
    return success();                                              \
  }

#define BINARY_EMIT(OPTYPE, FUNCNAME)                              \
  LogicalResult SimFHEEmitter::printOperation(OPTYPE op) {         \
    os << "stats += evaluator." FUNCNAME "("                       \
       << getName(op.getOperands().front()) << ", arch_params)\n"; \
    return success();                                              \
  }

BINARY_EMIT(ckks::AddOp, "add");
BINARY_EMIT(ckks::AddPlainOp, "add_plain");
BINARY_EMIT(ckks::SubOp, "subtract");
BINARY_EMIT(ckks::SubPlainOp, "subtract_plain");
BINARY_EMIT(ckks::MulOp, "multiply");
BINARY_EMIT(ckks::MulPlainOp, "multiply_plain");
UNARY_EMIT(ckks::NegateOp, "negate");
UNARY_EMIT(ckks::RotateOp, "rotate");
UNARY_EMIT(ckks::RelinearizeOp, "key_switch");
UNARY_EMIT(ckks::RescaleOp, "mod_reduce_rescale");
UNARY_EMIT(ckks::LevelReduceOp, "mod_down_reduce");
UNARY_EMIT(ckks::BootstrapOp, "bootstrap");

#undef UNARY_EMIT
#undef BINARY_EMIT

}  // namespace simfhe
}  // namespace heir
}  // namespace mlir
