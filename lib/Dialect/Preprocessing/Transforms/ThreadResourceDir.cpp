#include "lib/Dialect/Preprocessing/Transforms/ThreadResourceDir.h"

#include "lib/Dialect/Preprocessing/IR/PreprocessingOps.h"
#include "lib/Dialect/Preprocessing/IR/PreprocessingTypes.h"
#include "llvm/include/llvm/ADT/SetVector.h"            // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project

namespace mlir {
namespace heir {
namespace preprocessing {

#define GEN_PASS_DEF_THREADRESOURCEDIR
#include "lib/Dialect/Preprocessing/Transforms/Passes.h.inc"

namespace {

struct ThreadResourceDir : impl::ThreadResourceDirBase<ThreadResourceDir> {
  using ThreadResourceDirBase::ThreadResourceDirBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext* ctx = &getContext();
    auto callee = [&](func::CallOp call) {
      return module.lookupSymbol<func::FuncOp>(call.getCallee());
    };

    // Functions that load a resource, closed over their callers.
    SetVector<func::FuncOp> needsDirectory;
    module.walk([&](LoadResourceOp load) {
      if (!load.getDirectory())
        needsDirectory.insert(load->getParentOfType<func::FuncOp>());
    });
    bool changed = true;
    while (changed) {
      changed = false;
      module.walk([&](func::CallOp call) {
        func::FuncOp target = callee(call);
        auto caller = call->getParentOfType<func::FuncOp>();
        if (target && caller && needsDirectory.contains(target))
          changed |= needsDirectory.insert(caller);
      });
    }

    Type directoryType = ResourceDirType::get(ctx);
    for (func::FuncOp function : needsDirectory) {
      unsigned index = function.getNumArguments();
      if (failed(function.insertArgument(index, directoryType, DictionaryAttr(),
                                         function.getLoc())))
        return signalPassFailure();
      Value directory = function.getArgument(index);
      function.walk([&](LoadResourceOp load) {
        if (!load.getDirectory()) load.getDirectoryMutable().assign(directory);
      });
    }
    WalkResult result = module.walk([&](func::CallOp call) {
      func::FuncOp target = callee(call);
      if (!target || !needsDirectory.contains(target))
        return WalkResult::advance();
      auto caller = call->getParentOfType<func::FuncOp>();
      if (!caller) {
        call.emitOpError("calls a resource-loading function from outside a "
                         "function, so no resource directory reaches it");
        return WalkResult::interrupt();
      }
      call->insertOperands(call.getNumOperands(), caller.getArguments().back());
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) return signalPassFailure();
  }
};

}  // namespace

}  // namespace preprocessing
}  // namespace heir
}  // namespace mlir
