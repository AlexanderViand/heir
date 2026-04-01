#include "lib/Dialect/Cheddar/Transforms/ConfigureCryptoContext.h"

#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/Cheddar/IR/CheddarAttributes.h"
#include "lib/Dialect/Cheddar/IR/CheddarDialect.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"         // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"       // from @llvm-project

namespace mlir::heir::cheddar {

#define GEN_PASS_DEF_CHEDDARCONFIGURECRYPTOCONTEXT
#include "lib/Dialect/Cheddar/Transforms/ConfigureCryptoContext.h.inc"

struct CheddarConfigureCryptoContext
    : public impl::CheddarConfigureCryptoContextBase<
          CheddarConfigureCryptoContext> {
  void runOnOperation() override {
    auto moduleOp = getOperation();
    MLIRContext *ctx = &getContext();

    // Read CKKS scheme params and convert to CHEDDAR attrs
    auto schemeParamAttr = moduleOp->getAttrOfType<ckks::SchemeParamAttr>(
        ckks::CKKSDialect::kSchemeParamAttrName);

    if (schemeParamAttr) {
      // Store scheme params as CHEDDAR-specific module attributes
      // The emitter reads these to generate parameter construction code.
      moduleOp->setAttr("cheddar.logN",
                        IntegerAttr::get(IntegerType::get(ctx, 64),
                                         schemeParamAttr.getLogN()));
      moduleOp->setAttr("cheddar.logDefaultScale",
                        IntegerAttr::get(IntegerType::get(ctx, 64),
                                         schemeParamAttr.getLogDefaultScale()));
      if (auto Q = schemeParamAttr.getQ()) {
        moduleOp->setAttr("cheddar.Q", Q);
      }
      if (auto P = schemeParamAttr.getP()) {
        moduleOp->setAttr("cheddar.P", P);
      }

      // Remove the CKKS scheme param attribute — consumed
      moduleOp->removeAttr(ckks::CKKSDialect::kSchemeParamAttrName);
    }

    // Remove scheme.ckks marker attribute
    moduleOp->removeAttr("scheme.ckks");
  }
};

}  // namespace mlir::heir::cheddar
