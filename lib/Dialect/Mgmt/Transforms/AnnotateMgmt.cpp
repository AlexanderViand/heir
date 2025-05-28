#include "lib/Dialect/Mgmt/Transforms/AnnotateMgmt.h"

#include "lib/Analysis/DimensionAnalysis/DimensionAnalysis.h"
#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/ScaleAnalysis/ScaleAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "lib/Dialect/Mgmt/Transforms/Utils.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Utils/AttributeUtils.h"
#include "lib/Utils/Utils.h"
#include "mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/SymbolTable.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {
namespace mgmt {

#define GEN_PASS_DEF_ANNOTATEMGMT
#include "lib/Dialect/Mgmt/Transforms/Passes.h.inc"

void annotateMgmtAttr(Operation *top) {
  auto mergeIntoMgmtAttr = [&](Attribute levelAttr, Attribute dimensionAttr,
                               Attribute scaleAttr) {
    auto level = cast<IntegerAttr>(levelAttr).getInt();
    auto dimension = cast<IntegerAttr>(dimensionAttr).getInt();
    if (!scaleAttr) {
      return MgmtAttr::get(top->getContext(), level, dimension);
    }
    auto scale = cast<IntegerAttr>(scaleAttr).getInt();
    return MgmtAttr::get(top->getContext(), level, dimension, scale);
  };

  walkValues(top, [&](Value value) {
    if (succeeded(findAttributeAssociatedWith(value,
                                              MgmtDialect::kArgMgmtAttrName))) {
      return;
    }

    FailureOr<Attribute> levelAttr =
        findAttributeAssociatedWith(value, kArgLevelAttrName);
    FailureOr<Attribute> dimensionAttr =
        findAttributeAssociatedWith(value, kArgDimensionAttrName);
    if (failed(levelAttr) || failed(dimensionAttr)) {
      return;
    }
    Attribute scaleAttr =
        findAttributeAssociatedWith(value, kArgScaleAttrName).value_or(nullptr);
    Attribute mgmtAttr =
        mergeIntoMgmtAttr(levelAttr.value(), dimensionAttr.value(), scaleAttr);
    setAttributeAssociatedWith(value, MgmtDialect::kArgMgmtAttrName, mgmtAttr);
  });

  // Function declarations have no values and must be handled manually
  top->walk<WalkOrder::PreOrder>([&](func::FuncOp funcOp) {
    if (!funcOp.isExternal()) return;
    for (auto i = 0; i != funcOp.getNumArguments(); ++i) {
      auto argumentTy = funcOp.getFunctionType().getInput(i);
      if (isa<secret::SecretType>(argumentTy)) {
        funcOp.setArgAttr(i, MgmtDialect::kArgMgmtAttrName,
                          MgmtAttr::get(top->getContext(), 0, 2));
      }
    }
  });
}

struct AnnotateMgmt : impl::AnnotateMgmtBase<AnnotateMgmt> {
  using AnnotateMgmtBase::AnnotateMgmtBase;

  void runOnOperation() override {
    DataFlowSolver solver;
    SymbolTableCollection symbolTable;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<SecretnessAnalysis>();
    solver.load<LevelAnalysis>();
    solver.load<LevelAnalysisBackward>(symbolTable);
    solver.load<DimensionAnalysis>();

    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    clearAttrs(getOperation(), MgmtDialect::kArgMgmtAttrName);
    annotateLevel(getOperation(), &solver, baseLevel);
    annotateDimension(getOperation(), &solver);
    // Combine level and dimension (and optional scale) into MgmtAttr
    // also removes the level/dimension/(optional scale) annotations.
    // The optional scale is passed by annotateScale() when calling
    // this pass.
    annotateMgmtAttr(getOperation());

    clearAttrs(getOperation(), kArgLevelAttrName);
    clearAttrs(getOperation(), kArgDimensionAttrName);
    clearAttrs(getOperation(), kArgScaleAttrName);

    // Dataflow analyses don't assign anything to function results because they
    // don't have a corresponding Value. So we have to manually copy it from
    // the func terminator.
    copyReturnOperandAttrsToFuncResultAttrs(getOperation(),
                                            MgmtDialect::kArgMgmtAttrName);

    if (failed(copyMgmtAttrToClientHelpers(getOperation()))) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace mgmt
}  // namespace heir
}  // namespace mlir
