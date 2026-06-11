#include "lib/Transforms/ModArithToArithVeir/ModArithToArithVeir.h"

#include <cstdlib>
#include <optional>
#include <string>
#include <system_error>

#include "llvm/include/llvm/ADT/SmallString.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"          // from @llvm-project
#include "llvm/include/llvm/Support/ErrorOr.h"        // from @llvm-project
#include "llvm/include/llvm/Support/FileSystem.h"     // from @llvm-project
#include "llvm/include/llvm/Support/FileUtilities.h"  // from @llvm-project
#include "llvm/include/llvm/Support/MemoryBuffer.h"   // from @llvm-project
#include "llvm/include/llvm/Support/Program.h"        // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"           // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"     // from @llvm-project
#include "mlir/include/mlir/IR/OwningOpRef.h"          // from @llvm-project
#include "mlir/include/mlir/Parser/Parser.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"            // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_MODARITHTOARITHVEIR
#include "lib/Transforms/ModArithToArithVeir/ModArithToArithVeir.h.inc"

namespace {

// Locates the veir-opt binary, in order of precedence: the pass option, the
// HEIR_VEIR_OPT_PATH environment variable, and finally the PATH.
std::string findVeirOpt(const std::string& veirOptPath) {
  if (!veirOptPath.empty()) return veirOptPath;
  if (const char* envPath = std::getenv("HEIR_VEIR_OPT_PATH"))
    if (envPath[0] != '\0') return envPath;
  if (llvm::ErrorOr<std::string> result =
          llvm::sys::findProgramByName("veir-opt"))
    return *result;
  return "";
}

}  // namespace

struct ModArithToArithVeir
    : impl::ModArithToArithVeirBase<ModArithToArithVeir> {
  using ModArithToArithVeirBase::ModArithToArithVeirBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    std::string veirOpt = findVeirOpt(veirOptPath);
    if (veirOpt.empty() || !llvm::sys::fs::can_execute(veirOpt)) {
      module.emitError()
          << "could not find an executable veir-opt binary; set the "
             "veir-opt-path pass option or the HEIR_VEIR_OPT_PATH environment "
             "variable, or add veir-opt to the PATH";
      return signalPassFailure();
    }

    // Print the module in the generic op form to a temporary file. VEIR only
    // understands the generic op syntax and local-scope (inline) aliases;
    // locations are not printed (and hence dropped).
    SmallString<128> inputPath;
    int inputFd;
    if (std::error_code ec = llvm::sys::fs::createTemporaryFile(
            "heir-veir-input", "mlir", inputFd, inputPath)) {
      module.emitError() << "failed to create temporary input file: "
                         << ec.message();
      return signalPassFailure();
    }
    llvm::FileRemover inputRemover(inputPath);
    {
      llvm::raw_fd_ostream inputStream(inputFd, /*shouldClose=*/true);
      module.print(inputStream,
                   OpPrintingFlags().printGenericOpForm().useLocalScope());
      inputStream << "\n";
    }

    SmallString<128> outputPath;
    if (std::error_code ec = llvm::sys::fs::createTemporaryFile(
            "heir-veir-output", "mlir", outputPath)) {
      module.emitError() << "failed to create temporary output file: "
                         << ec.message();
      return signalPassFailure();
    }
    llvm::FileRemover outputRemover(outputPath);

    SmallString<128> stderrPath;
    if (std::error_code ec = llvm::sys::fs::createTemporaryFile(
            "heir-veir-stderr", "txt", stderrPath)) {
      module.emitError() << "failed to create temporary stderr file: "
                         << ec.message();
      return signalPassFailure();
    }
    llvm::FileRemover stderrRemover(stderrPath);

    // Run veir-opt. The pipeline must be passed as a single argv element.
    std::string pipelineArg = "-p=" + pipeline;
    SmallVector<StringRef, 3> args = {veirOpt, inputPath, pipelineArg};
    std::optional<StringRef> redirects[] = {
        /*stdin=*/std::nullopt, /*stdout=*/outputPath.str(),
        /*stderr=*/stderrPath.str()};
    std::string errorMessage;
    int exitCode =
        llvm::sys::ExecuteAndWait(veirOpt, args, /*Env=*/std::nullopt,
                                  redirects, /*SecondsToWait=*/0,
                                  /*MemoryLimit=*/0, &errorMessage);
    if (exitCode != 0) {
      auto diag = module.emitError()
                  << "veir-opt invocation '" << veirOpt << " " << inputPath
                  << " " << pipelineArg << "' failed with exit code "
                  << exitCode;
      if (!errorMessage.empty()) diag << ": " << errorMessage;
      if (auto stderrBuffer = llvm::MemoryBuffer::getFile(stderrPath))
        if (!(*stderrBuffer)->getBuffer().empty())
          diag << "; stderr:\n" << (*stderrBuffer)->getBuffer();
      return signalPassFailure();
    }

    // Read back the output and re-parse it in this pass's context.
    auto outputBuffer = llvm::MemoryBuffer::getFile(outputPath);
    if (!outputBuffer) {
      module.emitError() << "failed to read veir-opt output file "
                         << outputPath << ": "
                         << outputBuffer.getError().message();
      return signalPassFailure();
    }
    StringRef output = (*outputBuffer)->getBuffer();

    ParserConfig config(&getContext());
    OwningOpRef<ModuleOp> parsed =
        parseSourceString<ModuleOp>(output, config, /*sourceName=*/outputPath);
    if (!parsed) {
      module.emitError() << "failed to parse veir-opt output back into MLIR; "
                            "raw output was:\n"
                         << output;
      return signalPassFailure();
    }

    // Replace the contents of the current module with the parsed result,
    // keeping the original module op (and its attributes) intact.
    Block* body = module.getBody();
    while (!body->empty()) body->front().erase();
    body->getOperations().splice(body->end(),
                                 parsed->getBody()->getOperations());
  }
};

}  // namespace heir
}  // namespace mlir
