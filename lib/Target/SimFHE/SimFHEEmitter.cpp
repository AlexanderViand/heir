#include "lib/Target/SimFHE/SimFHEEmitter.h"

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"      // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"  // from @llvm-project
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
        registry.insert<func::FuncDialect, ckks::CKKSDialect>();
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
  os << commaSeparatedValues(funcOp.getArguments(),
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

#define SIMPLE_EMIT(OPTYPE, FUNCNAME)                               \
  LogicalResult SimFHEEmitter::printOperation(OPTYPE op) {          \
    os << "evaluator." FUNCNAME "(" << getName(op.getOperands()[0]) \
       << ", arch_params)\n";                                       \
    return success();                                               \
  }

SIMPLE_EMIT(ckks::AddOp, "add");
SIMPLE_EMIT(ckks::AddPlainOp, "add_plain");
SIMPLE_EMIT(ckks::SubOp, "add");
SIMPLE_EMIT(ckks::SubPlainOp, "add_plain");
SIMPLE_EMIT(ckks::MulOp, "multiply");
SIMPLE_EMIT(ckks::MulPlainOp, "multiply_plain");
SIMPLE_EMIT(ckks::NegateOp, "negate");
SIMPLE_EMIT(ckks::RotateOp, "rotate");
SIMPLE_EMIT(ckks::RelinearizeOp, "key_switch");
SIMPLE_EMIT(ckks::RescaleOp, "mod_reduce_rescale");
SIMPLE_EMIT(ckks::LevelReduceOp, "mod_down_reduce");
SIMPLE_EMIT(ckks::BootstrapOp, "bootstrap");

#undef SIMPLE_EMIT

}  // namespace simfhe
}  // namespace heir
}  // namespace mlir
