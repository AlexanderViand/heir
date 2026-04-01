#include "lib/Dialect/Cheddar/Transforms/ConfigureCryptoContext.h"

#include <vector>

#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/Cheddar/IR/CheddarAttributes.h"
#include "lib/Dialect/Cheddar/IR/CheddarDialect.h"
#include "lib/Parameters/RLWEParams.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"         // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"       // from @llvm-project

namespace mlir::heir::cheddar {

// CHEDDAR's GPU kernels require a minimum number of RNS limbs to function
// correctly. Shallow computations (depth 1-2) generate too few primes for
// CHEDDAR's NTT and key-switch GPU kernels. Pad with extra scaling primes.
// CHEDDAR's GPU kernels may require a minimum number of RNS limbs.
// For now, don't pad — shallow computations must produce enough primes
// through the scheme management. The minimum viable CHEDDAR parameter
// set appears to be ~4 Q primes based on empirical testing.
// TODO: determine the actual CHEDDAR minimum and pad if needed.
static constexpr int kMinQPrimes = 0;
static constexpr int kMinPPrimes = 0;

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
      int64_t logN = schemeParamAttr.getLogN();
      int64_t logDefaultScale = schemeParamAttr.getLogDefaultScale();
      int ringDim = 1 << logN;

      moduleOp->setAttr("cheddar.logN",
                        IntegerAttr::get(IntegerType::get(ctx, 64), logN));
      moduleOp->setAttr(
          "cheddar.logDefaultScale",
          IntegerAttr::get(IntegerType::get(ctx, 64), logDefaultScale));

      // Pad Q primes if below CHEDDAR's minimum
      if (auto Q = schemeParamAttr.getQ()) {
        auto qVals = Q.asArrayRef();
        if (static_cast<int>(qVals.size()) < kMinQPrimes) {
          std::vector<int64_t> paddedQ(qVals.begin(), qVals.end());
          // First prime is the "first mod" (larger), rest are scaling mods.
          // Pad with additional scaling-mod-sized primes.
          while (static_cast<int>(paddedQ.size()) < kMinQPrimes) {
            paddedQ.push_back(findPrime(logDefaultScale, ringDim, paddedQ));
          }
          moduleOp->setAttr("cheddar.Q", DenseI64ArrayAttr::get(ctx, paddedQ));
        } else {
          moduleOp->setAttr("cheddar.Q", Q);
        }
      }

      // Pad P primes if below minimum
      if (auto P = schemeParamAttr.getP()) {
        auto pVals = P.asArrayRef();
        if (static_cast<int>(pVals.size()) < kMinPPrimes) {
          std::vector<int64_t> paddedP(pVals.begin(), pVals.end());
          // P primes are larger (typically 60 bits for 64-bit word)
          int pBits = 60;
          // Collect all existing primes to avoid collisions
          std::vector<int64_t> allPrimes(paddedP.begin(), paddedP.end());
          if (auto Q = schemeParamAttr.getQ()) {
            auto qVals = Q.asArrayRef();
            allPrimes.insert(allPrimes.end(), qVals.begin(), qVals.end());
          }
          while (static_cast<int>(paddedP.size()) < kMinPPrimes) {
            int64_t p = findPrime(pBits, ringDim, allPrimes);
            paddedP.push_back(p);
            allPrimes.push_back(p);
          }
          moduleOp->setAttr("cheddar.P", DenseI64ArrayAttr::get(ctx, paddedP));
        } else {
          moduleOp->setAttr("cheddar.P", P);
        }
      }

      // Remove the CKKS scheme param attribute — consumed
      moduleOp->removeAttr(ckks::CKKSDialect::kSchemeParamAttrName);
    }

    // Remove scheme.ckks marker attribute
    moduleOp->removeAttr("scheme.ckks");
  }
};

}  // namespace mlir::heir::cheddar
