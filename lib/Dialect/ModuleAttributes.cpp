#include "lib/Dialect/ModuleAttributes.h"

#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "mlir/include/mlir/IR/Attributes.h"         // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"          // from @llvm-project

namespace mlir {
namespace heir {

/*===----------------------------------------------------------------------===*/
// Module Attributes for Scheme
/*===----------------------------------------------------------------------===*/

bool moduleIsBGV(Operation* moduleOp) {
  return moduleOp->getAttrOfType<mlir::UnitAttr>(kBGVSchemeAttrName) != nullptr;
}

bool moduleIsBFV(Operation* moduleOp) {
  return moduleOp->getAttrOfType<mlir::UnitAttr>(kBFVSchemeAttrName) != nullptr;
}

bool moduleIsBGVOrBFV(Operation* moduleOp) {
  return moduleIsBGV(moduleOp) || moduleIsBFV(moduleOp);
}

bool moduleIsCKKS(Operation* moduleOp) {
  return moduleOp->getAttrOfType<mlir::UnitAttr>(kCKKSSchemeAttrName) !=
         nullptr;
}

bool moduleIsCGGI(Operation* moduleOp) {
  return moduleOp->getAttrOfType<mlir::UnitAttr>(kCGGISchemeAttrName) !=
         nullptr;
}

Attribute getSchemeParamAttr(Operation* op) {
  SmallVector<StringLiteral> schemeAttrNames = {
      bgv::BGVDialect::kSchemeParamAttrName,
      ckks::CKKSDialect::kSchemeParamAttrName,
  };

  Operation* moduleOp = op;
  if (!isa<ModuleOp>(op)) {
    moduleOp = op->getParentOfType<ModuleOp>();
  }

  for (auto schemeAttrName : schemeAttrNames) {
    if (auto schemeAttr = moduleOp->getAttr(schemeAttrName)) {
      return schemeAttr;
    }
  }

  return UnitAttr::get(op->getContext());
}

void moduleClearScheme(Operation* moduleOp) {
  moduleOp->removeAttr(kBGVSchemeAttrName);
  moduleOp->removeAttr(kBFVSchemeAttrName);
  moduleOp->removeAttr(kCKKSSchemeAttrName);
  moduleOp->removeAttr(kCGGISchemeAttrName);
}

void moduleSetBGV(Operation* moduleOp) {
  moduleClearScheme(moduleOp);
  moduleOp->setAttr(kBGVSchemeAttrName,
                    mlir::UnitAttr::get(moduleOp->getContext()));
}

void moduleSetBFV(Operation* moduleOp) {
  moduleClearScheme(moduleOp);
  moduleOp->setAttr(kBFVSchemeAttrName,
                    mlir::UnitAttr::get(moduleOp->getContext()));
}

void moduleSetCKKS(Operation* moduleOp) {
  moduleClearScheme(moduleOp);
  moduleOp->setAttr(kCKKSSchemeAttrName,
                    mlir::UnitAttr::get(moduleOp->getContext()));
}

void moduleSetCGGI(Operation* moduleOp) {
  moduleClearScheme(moduleOp);
  moduleOp->setAttr(kCGGISchemeAttrName,
                    mlir::UnitAttr::get(moduleOp->getContext()));
}

StringRef getModuleCKKSScalePolicy(Operation* moduleOp) {
  if (auto policy =
          moduleOp->getAttrOfType<mlir::StringAttr>(kCKKSScalePolicyAttrName)) {
    return policy.getValue();
  }
  return kCKKSNominalScalePolicyValue;
}

bool moduleUsesPreciseCKKSScalePolicy(Operation* moduleOp) {
  return getModuleCKKSScalePolicy(moduleOp) == kCKKSPreciseScalePolicyValue;
}

bool moduleUsesQAwarePreciseCKKSScalePolicy(Operation* moduleOp) {
  // CHEDDAR's CKKS path uses a backend-defined per-level scale schedule rather
  // than the generic q_i-aware exact rescale model. Keep "precise" symbolic
  // for that backend until it grows a dedicated backend-aware realization.
  return moduleUsesPreciseCKKSScalePolicy(moduleOp) &&
         !moduleIsCheddar(moduleOp);
}

void moduleSetCKKSScalePolicy(Operation* moduleOp, StringRef policy) {
  moduleOp->setAttr(kCKKSScalePolicyAttrName,
                    mlir::StringAttr::get(moduleOp->getContext(), policy));
}

/*===----------------------------------------------------------------------===*/
// Module Attributes for Backend
/*===----------------------------------------------------------------------===*/

bool moduleIsOpenfhe(Operation* moduleOp) {
  return moduleOp->getAttrOfType<mlir::UnitAttr>(kOpenfheBackendAttrName) !=
         nullptr;
}

bool moduleIsLattigo(Operation* moduleOp) {
  return moduleOp->getAttrOfType<mlir::UnitAttr>(kLattigoBackendAttrName) !=
         nullptr;
}

bool moduleIsCheddar(Operation* moduleOp) {
  return moduleOp->getAttrOfType<mlir::UnitAttr>(kCheddarBackendAttrName) !=
         nullptr;
}

void moduleClearBackend(Operation* moduleOp) {
  moduleOp->removeAttr(kOpenfheBackendAttrName);
  moduleOp->removeAttr(kLattigoBackendAttrName);
  moduleOp->removeAttr(kCheddarBackendAttrName);
}

void moduleSetOpenfhe(Operation* moduleOp) {
  moduleClearBackend(moduleOp);
  moduleOp->setAttr(kOpenfheBackendAttrName,
                    mlir::UnitAttr::get(moduleOp->getContext()));
}

void moduleSetLattigo(Operation* moduleOp) {
  moduleClearBackend(moduleOp);
  moduleOp->setAttr(kLattigoBackendAttrName,
                    mlir::UnitAttr::get(moduleOp->getContext()));
}

void moduleSetCheddar(Operation* moduleOp) {
  moduleClearBackend(moduleOp);
  moduleOp->setAttr(kCheddarBackendAttrName,
                    mlir::UnitAttr::get(moduleOp->getContext()));
}

}  // namespace heir
}  // namespace mlir
