#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/CGGI/IR/CGGIDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/Comb/IR/CombDialect.h"
#include "lib/Dialect/Debug/IR/DebugDialect.h"
#include "lib/Dialect/Jaxite/IR/JaxiteDialect.h"
#include "lib/Dialect/JaxiteWord/IR/JaxiteWordDialect.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/Lattigo/IR/LattigoDialect.h"
#include "lib/Dialect/MathExt/IR/MathExtDialect.h"
#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "lib/Dialect/ModArith/IR/ModArithDialect.h"
#include "lib/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "lib/Dialect/Orion/IR/OrionDialect.h"
#include "lib/Dialect/Polynomial/IR/PolynomialDialect.h"
#include "lib/Dialect/RNS/IR/RNSDialect.h"
#include "lib/Dialect/RNS/IR/RNSTypeInterfaces.h"
#include "lib/Dialect/Random/IR/RandomDialect.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TfheRust/IR/TfheRustDialect.h"
#include "lib/Dialect/TfheRustBool/IR/TfheRustBoolDialect.h"
#include "mlir/include/mlir/InitAllDialects.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"  // from @llvm-project

using namespace mlir;
using namespace heir;

int main(int argc, char** argv) {
  mlir::DialectRegistry registry;

  registry.insert<bgv::BGVDialect>();
  registry.insert<ckks::CKKSDialect>();
  registry.insert<cggi::CGGIDialect>();
  registry.insert<comb::CombDialect>();
  registry.insert<debug::DebugDialect>();
  registry.insert<jaxite::JaxiteDialect>();
  registry.insert<jaxiteword::JaxiteWordDialect>();
  registry.insert<lattigo::LattigoDialect>();
  registry.insert<lwe::LWEDialect>();
  registry.insert<mod_arith::ModArithDialect>();
  registry.insert<mgmt::MgmtDialect>();
  registry.insert<random::RandomDialect>();
  registry.insert<openfhe::OpenfheDialect>();
  registry.insert<orion::OrionDialect>();
  registry.insert<rns::RNSDialect>();
  registry.insert<secret::SecretDialect>();
  registry.insert<tensor_ext::TensorExtDialect>();
  registry.insert<tfhe_rust::TfheRustDialect>();
  registry.insert<tfhe_rust_bool::TfheRustBoolDialect>();
  registry.insert<math_ext::MathExtDialect>();

  rns::registerExternalRNSTypeInterfaces(registry);

  // Keep the upstream MLIR dialect surface available in the LSP, including
  // GPU-related dialects used by HEIR's lowering flows.
  registerAllDialects(registry);

  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
