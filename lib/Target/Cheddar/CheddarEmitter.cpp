#include "lib/Target/Cheddar/CheddarEmitter.h"

#include <cmath>
#include <string>

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/Cheddar/IR/CheddarDialect.h"
#include "lib/Dialect/Cheddar/IR/CheddarOps.h"
#include "lib/Dialect/Cheddar/IR/CheddarTypes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Target/Cheddar/CheddarTemplates.h"
#include "lib/Utils/TargetUtils.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"         // from @llvm-project
#include "llvm/include/llvm/Support/ManagedStatic.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/ReshapeOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"   // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"          // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"               // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar {

CheddarEmitter::CheddarEmitter(raw_ostream &os,
                               SelectVariableNames *variableNames,
                               bool use64Bit, const std::string &paramsJsonPath)
    : os(os),
      variableNames(variableNames),
      use64Bit(use64Bit),
      paramsJsonPath(paramsJsonPath) {}

std::string CheddarEmitter::getName(Value value) {
  return variableNames->getNameForValue(value);
}

FailureOr<std::string> CheddarEmitter::convertType(Type type, bool asArg) {
  return llvm::TypeSwitch<Type, FailureOr<std::string>>(type)
      .Case<ContextType>([](auto) { return std::string("CtxPtr"); })
      .Case<ParameterType>([asArg](auto) {
        return std::string(asArg ? "const Param&" : "Param");
      })
      .Case<EncoderType>([](auto) { return std::string("Enc&"); })
      .Case<UserInterfaceType>([](auto) { return std::string("UI&"); })
      // CHEDDAR containers are move-only: pass by const ref, return by value
      .Case<CiphertextType>(
          [asArg](auto) { return std::string(asArg ? "const Ct&" : "Ct"); })
      .Case<PlaintextType>(
          [asArg](auto) { return std::string(asArg ? "const Pt&" : "Pt"); })
      .Case<ConstantType>([asArg](auto) {
        return std::string(asArg ? "const Const&" : "Const");
      })
      .Case<EvalKeyType>([](auto) { return std::string("const Evk&"); })
      .Case<EvkMapType>([](auto) { return std::string("const EvkMapT&"); })
      .Case<RankedTensorType>(
          [asArg](RankedTensorType type) -> FailureOr<std::string> {
            auto elemType = type.getElementType();
            if (isa<ComplexType>(elemType)) {
              return std::string(asArg ? "const std::vector<Complex>&"
                                       : "std::vector<Complex>");
            }
            if (elemType.isF64() || elemType.isF32()) {
              return std::string(asArg ? "const std::vector<double>&"
                                       : "std::vector<double>");
            }
            if (elemType.isInteger(32) || elemType.isInteger(64) ||
                elemType.isIndex()) {
              return std::string(asArg ? "const std::vector<int64_t>&"
                                       : "std::vector<int64_t>");
            }
            if (isa<CiphertextType>(elemType)) {
              return std::string(asArg ? "const std::vector<Ct>&"
                                       : "std::vector<Ct>");
            }
            if (isa<PlaintextType>(elemType)) {
              return std::string(asArg ? "const std::vector<Pt>&"
                                       : "std::vector<Pt>");
            }
            return failure();
          })
      .Case<FloatType>([](auto) { return std::string("double"); })
      .Case<IndexType>([](auto) { return std::string("int64_t"); })
      .Case<IntegerType>([](IntegerType type) -> FailureOr<std::string> {
        auto width = type.getWidth();
        if (width == 1) return std::string("bool");
        if (width <= 32) return std::string("int32_t");
        return std::string("int64_t");
      })
      .Default([](Type) { return failure(); });
}

void CheddarEmitter::emitPrelude(raw_ostream &os) const {
  os << kStdIncludes;
  os << kCheddarInclude;
  if (needsExtensionIncludes) {
    os << kCheddarExtensionInclude;
  }
  if (needsJsonIncludes) {
    os << kJsonInclude;
  }
  os << "\n";
  if (use64Bit) {
    os << kTypeAliasPrelude64;
  } else {
    os << kTypeAliasPrelude32;
  }
  os << "\n";
}

LogicalResult CheddarEmitter::translate(Operation &op) {
  LogicalResult status =
      llvm::TypeSwitch<Operation &, LogicalResult>(op)
          .Case<ModuleOp>([&](auto op) { return printOperation(op); })
          .Case<func::FuncOp, func::ReturnOp, func::CallOp>(
              [&](auto op) { return printOperation(op); })
          .Case<arith::ConstantOp>([&](auto op) { return printOperation(op); })
          // Cheddar setup ops
          .Case<CreateContextOp, CreateUserInterfaceOp, GetEncoderOp,
                GetEvkMapOp, GetMultKeyOp, GetRotKeyOp, GetConjKeyOp,
                PrepareRotKeyOp>([&](auto op) { return printOperation(op); })
          // Encode/encrypt/decrypt
          .Case<EncodeOp, EncodeConstantOp, DecodeOp, EncryptOp, DecryptOp>(
              [&](auto op) { return printOperation(op); })
          // Ct-ct arithmetic
          .Case<AddOp, SubOp, MultOp, NegOp>(
              [&](auto op) { return printOperation(op); })
          // Ct-pt and ct-const
          .Case<AddPlainOp, SubPlainOp, MultPlainOp, AddConstOp, MultConstOp>(
              [&](auto op) { return printOperation(op); })
          // Unary / level management
          .Case<RescaleOp, LevelDownOp, RelinearizeOp, RelinearizeRescaleOp>(
              [&](auto op) { return printOperation(op); })
          // Fused compound ops
          .Case<HMultOp, HRotOp, HRotAddOp, HConjOp, HConjAddOp, MadUnsafeOp>(
              [&](auto op) { return printOperation(op); })
          // Extension ops
          .Case<BootOp, LinearTransformOp, EvalPolyOp>(
              [&](auto op) { return printOperation(op); })
          // SCF control flow
          .Case<scf::ForOp, scf::IfOp, scf::YieldOp>(
              [&](auto op) { return printOperation(op); })
          // Tensor ops
          .Case<tensor::EmptyOp, tensor::ExtractOp, tensor::InsertOp,
                tensor::FromElementsOp, tensor::ExpandShapeOp,
                tensor::ExtractSliceOp, tensor::InsertSliceOp>(
              [&](auto op) { return printOperation(op); })
          // Additional arith ops
          .Case<arith::AddIOp, arith::SubIOp, arith::RemSIOp, arith::CmpIOp,
                arith::IndexCastOp>([&](auto op) { return printOperation(op); })
          .Default([&](Operation &) {
            return op.emitOpError("unable to find printer for op");
          });
  return status;
}

//===----------------------------------------------------------------------===//
// Module / Function
//===----------------------------------------------------------------------===//

LogicalResult CheddarEmitter::printOperation(ModuleOp moduleOp) {
  // Read module-level scheme params set by ConfigureCryptoContext
  if (auto attr =
          moduleOp->getAttrOfType<IntegerAttr>("cheddar.logDefaultScale"))
    logDefaultScale = attr.getInt();

  for (Operation &op : moduleOp) {
    if (failed(translate(op))) return failure();
  }
  return success();
}

LogicalResult CheddarEmitter::printOperation(func::FuncOp funcOp) {
  // Emit function signature
  auto funcType = funcOp.getFunctionType();
  auto resultTypes = funcType.getResults();

  // Return type
  if (resultTypes.empty()) {
    os << "void ";
  } else if (resultTypes.size() == 1) {
    auto typeStr = convertType(resultTypes[0]);
    if (failed(typeStr))
      return funcOp.emitOpError("failed to convert return type");
    os << *typeStr << " ";
  } else {
    // Multiple returns: use std::tuple
    os << "std::tuple<";
    for (unsigned i = 0; i < resultTypes.size(); ++i) {
      if (i > 0) os << ", ";
      auto typeStr = convertType(resultTypes[i]);
      if (failed(typeStr))
        return funcOp.emitOpError("failed to convert return type");
      os << *typeStr;
    }
    os << "> ";
  }

  // Function name and arguments
  os << funcOp.getName() << "(";
  auto argTypes = funcType.getInputs();
  auto &entryBlock = funcOp.getBody().front();
  for (unsigned i = 0; i < argTypes.size(); ++i) {
    if (i > 0) os << ", ";
    auto typeStr = convertType(argTypes[i], /*asArg=*/true);
    if (failed(typeStr))
      return funcOp.emitOpError("failed to convert argument type");
    os << *typeStr << " " << getName(entryBlock.getArgument(i));
  }
  os << ") {\n";
  os.indent();

  // Emit body
  for (Block &block : funcOp.getBody()) {
    for (Operation &op : block.getOperations()) {
      if (failed(translate(op))) return failure();
    }
  }

  os.unindent();
  os << "}\n\n";
  return success();
}

LogicalResult CheddarEmitter::printOperation(func::ReturnOp op) {
  if (op.getNumOperands() == 0) {
    os << "return;\n";
  } else if (op.getNumOperands() == 1) {
    os << "return " << getName(op.getOperand(0)) << ";\n";
  } else {
    os << "return std::make_tuple(";
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      if (i > 0) os << ", ";
      os << "std::move(" << getName(op.getOperand(i)) << ")";
    }
    os << ");\n";
  }
  return success();
}

LogicalResult CheddarEmitter::printOperation(func::CallOp op) {
  // Declare results
  for (unsigned i = 0; i < op.getNumResults(); ++i) {
    auto typeStr = convertType(op.getResult(i).getType());
    if (failed(typeStr)) return op.emitOpError("failed to convert result type");
    os << *typeStr << " " << getName(op.getResult(i));
    if (i < op.getNumResults() - 1) os << ", ";
  }
  if (op.getNumResults() > 0) os << " = ";

  os << op.getCallee() << "(";
  for (unsigned i = 0; i < op.getNumOperands(); ++i) {
    if (i > 0) os << ", ";
    os << getName(op.getOperand(i));
  }
  os << ");\n";
  return success();
}

//===----------------------------------------------------------------------===//
// Arith ops
//===----------------------------------------------------------------------===//

LogicalResult CheddarEmitter::printOperation(arith::ConstantOp op) {
  auto result = op.getResult();
  auto attr = op.getValue();

  if (auto denseAttr = dyn_cast<DenseFPElementsAttr>(attr)) {
    auto name = getName(result);
    auto typeStr = convertType(result.getType());
    if (failed(typeStr))
      return op.emitOpError("failed to convert dense constant type");
    os << *typeStr << " " << name << " = {";
    bool first = true;
    for (auto val : denseAttr.getValues<APFloat>()) {
      if (!first) os << ", ";
      first = false;
      SmallString<16> str;
      val.toString(str, /*FormatPrecision=*/15);
      os << str;
    }
    os << "};\n";
    return success();
  }

  if (auto floatAttr = dyn_cast<FloatAttr>(attr)) {
    SmallString<16> str;
    floatAttr.getValue().toString(str, /*FormatPrecision=*/15);
    os << "double " << getName(result) << " = " << str << ";\n";
    return success();
  }

  if (auto denseIntAttr = dyn_cast<DenseIntElementsAttr>(attr)) {
    auto name = getName(result);
    auto typeStr = convertType(result.getType());
    if (failed(typeStr))
      return op.emitOpError("failed to convert dense int constant type");
    os << *typeStr << " " << name << " = {";
    bool first = true;
    for (auto val : denseIntAttr.getValues<APInt>()) {
      if (!first) os << ", ";
      first = false;
      os << val.getSExtValue();
    }
    os << "};\n";
    return success();
  }

  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    auto type = result.getType();
    auto typeStr = convertType(type);
    if (failed(typeStr))
      return op.emitOpError("failed to convert constant type");
    os << *typeStr << " " << getName(result) << " = "
       << intAttr.getValue().getSExtValue() << ";\n";
    return success();
  }

  return op.emitOpError("unsupported constant type for CHEDDAR emitter");
}

//===----------------------------------------------------------------------===//
// Setup ops
//===----------------------------------------------------------------------===//

LogicalResult CheddarEmitter::printOperation(CreateContextOp op) {
  os << "auto " << getName(op.getCtx()) << " = Context<word>::Create("
     << getName(op.getParams()) << ");\n";
  return success();
}

LogicalResult CheddarEmitter::printOperation(CreateUserInterfaceOp op) {
  os << "UI " << getName(op.getUi()) << "(" << getName(op.getCtx()) << ");\n";
  return success();
}

LogicalResult CheddarEmitter::printOperation(GetEncoderOp op) {
  os << "auto& " << getName(op.getEncoder()) << " = " << getName(op.getCtx())
     << "->encoder_;\n";
  return success();
}

LogicalResult CheddarEmitter::printOperation(GetEvkMapOp op) {
  os << "const auto& " << getName(op.getEvkMap()) << " = "
     << getName(op.getUi()) << ".GetEvkMap();\n";
  return success();
}

LogicalResult CheddarEmitter::printOperation(GetMultKeyOp op) {
  os << "const auto& " << getName(op.getKey()) << " = " << getName(op.getUi())
     << ".GetMultiplicationKey();\n";
  return success();
}

LogicalResult CheddarEmitter::printOperation(GetRotKeyOp op) {
  auto dist = op.getDistanceAttr().getInt();
  if (dist == -1) {
    // Sentinel for dynamic rotation: key lookup is inlined in HRotOp emitter.
    // Emit nothing; the HRotOp emitter traces back to the UI.
    return success();
  }
  os << "const auto& " << getName(op.getKey()) << " = " << getName(op.getUi())
     << ".GetRotationKey(" << dist << ");\n";
  return success();
}

LogicalResult CheddarEmitter::printOperation(GetConjKeyOp op) {
  os << "const auto& " << getName(op.getKey()) << " = " << getName(op.getUi())
     << ".GetConjugationKey();\n";
  return success();
}

LogicalResult CheddarEmitter::printOperation(PrepareRotKeyOp op) {
  auto dist = op.getDistanceAttr().getInt();
  auto maxLevel = op.getMaxLevelAttr().getInt();
  os << getName(op.getUi()) << ".PrepareRotationKey(" << dist << ", "
     << maxLevel << ");\n";
  return success();
}

//===----------------------------------------------------------------------===//
// Encode / Encrypt / Decrypt
//===----------------------------------------------------------------------===//

LogicalResult CheddarEmitter::printOperation(EncodeOp op) {
  auto level = op.getLevelAttr().getInt();
  auto logScale = op.getScaleAttr().getInt();
  auto name = getName(op.getPlaintext());
  auto msgName = getName(op.getMessage());

  // CHEDDAR's Encode takes std::vector<Complex>. If the message is
  // std::vector<double>, emit a conversion.
  auto msgType = op.getMessage().getType();
  bool needsComplexConversion = false;
  if (auto tensorType = dyn_cast<RankedTensorType>(msgType)) {
    auto elemType = tensorType.getElementType();
    needsComplexConversion = elemType.isF32() || elemType.isF64();
  }

  // Use CHEDDAR's Parameter::GetScale(level) for the correct runtime scale.
  // The logScale from the IR is the log2 of the nominal scale, but CHEDDAR's
  // actual scale per level depends on the prime chain. Using GetScale ensures
  // the plaintext scale matches the ciphertext scale exactly.
  //
  // For normal scale (logScale == logDefaultScale): use GetScale(level)
  // For doubled scale (logScale == 2*logDefaultScale, e.g. after mult
  //   before rescale): use GetScale(level)^2
  //
  // Find the context arg (first arg of the parent function) to access params.
  std::string ctxName;
  if (auto funcOp = op->getParentOfType<func::FuncOp>()) {
    if (funcOp.getNumArguments() > 0) ctxName = getName(funcOp.getArgument(0));
  }

  std::string scaleExpr;
  if (!ctxName.empty()) {
    std::string baseScale =
        ctxName + "->param_.GetScale(" + std::to_string(level) + ")";
    if (logScale > logDefaultScale) {
      // Doubled (or higher) scale: square the base scale per doubling
      int multiplier = logScale / logDefaultScale;
      scaleExpr = baseScale;
      for (int i = 1; i < multiplier; ++i) scaleExpr += " * " + baseScale;
    } else {
      scaleExpr = baseScale;
    }
  } else {
    // Fallback if context not found
    scaleExpr = "pow(2.0, " + std::to_string(logScale) + ")";
  }

  os << "Pt " << name << ";\n";
  if (needsComplexConversion) {
    std::string complexMsgName = name + "_complex";
    os << "std::vector<Complex> " << complexMsgName << "(" << msgName
       << ".begin(), " << msgName << ".end());\n";
    os << getName(op.getEncoder()) << ".Encode(" << name << ", " << level
       << ", " << scaleExpr << ", " << complexMsgName << ");\n";
  } else {
    os << getName(op.getEncoder()) << ".Encode(" << name << ", " << level
       << ", " << scaleExpr << ", " << msgName << ");\n";
  }
  return success();
}

LogicalResult CheddarEmitter::printOperation(EncodeConstantOp op) {
  auto level = op.getLevelAttr().getInt();
  auto logScale = op.getScaleAttr().getInt();
  auto name = getName(op.getConstant());
  os << "Const " << name << ";\n";
  os << getName(op.getEncoder()) << ".EncodeConstant(" << name << ", " << level
     << ", pow(2.0, " << logScale << "), " << getName(op.getValue()) << ");\n";
  return success();
}

LogicalResult CheddarEmitter::printOperation(DecodeOp op) {
  auto msgType = op.getMessage().getType();
  bool needsRealConversion = false;
  if (auto tensorType = dyn_cast<RankedTensorType>(msgType)) {
    auto elemType = tensorType.getElementType();
    needsRealConversion = elemType.isF32() || elemType.isF64();
  }

  if (needsRealConversion) {
    // CHEDDAR Decode returns Complex; convert to double for float tensors
    std::string complexName = getName(op.getMessage()) + "_complex";
    os << "std::vector<Complex> " << complexName << ";\n";
    os << getName(op.getEncoder()) << ".Decode(" << complexName << ", "
       << getName(op.getPlaintext()) << ");\n";
    os << "std::vector<double> " << getName(op.getMessage()) << "("
       << complexName << ".size());\n";
    os << "for (size_t i = 0; i < " << complexName << ".size(); ++i) "
       << getName(op.getMessage()) << "[i] = " << complexName
       << "[i].real();\n";
  } else {
    auto name = getName(op.getMessage());
    os << "std::vector<Complex> " << name << ";\n";
    os << getName(op.getEncoder()) << ".Decode(" << name << ", "
       << getName(op.getPlaintext()) << ");\n";
  }
  return success();
}

LogicalResult CheddarEmitter::printOperation(EncryptOp op) {
  auto name = getName(op.getCiphertext());
  os << "Ct " << name << ";\n";
  os << getName(op.getUi()) << ".Encrypt(" << name << ", "
     << getName(op.getPlaintext()) << ");\n";
  return success();
}

LogicalResult CheddarEmitter::printOperation(DecryptOp op) {
  auto name = getName(op.getPlaintext());
  os << "Pt " << name << ";\n";
  os << getName(op.getUi()) << ".Decrypt(" << name << ", "
     << getName(op.getCiphertext()) << ");\n";
  return success();
}

//===----------------------------------------------------------------------===//
// Binary ct-ct ops — CHEDDAR uses output-parameter style: ctx->Op(res, a, b)
//===----------------------------------------------------------------------===//

LogicalResult CheddarEmitter::printOperation(AddOp op) {
  auto name = getName(op.getOutput());
  os << "Ct " << name << ";\n";
  os << getName(op.getCtx()) << "->Add(" << name << ", " << getName(op.getLhs())
     << ", " << getName(op.getRhs()) << ");\n";
  return success();
}

LogicalResult CheddarEmitter::printOperation(SubOp op) {
  auto name = getName(op.getOutput());
  os << "Ct " << name << ";\n";
  os << getName(op.getCtx()) << "->Sub(" << name << ", " << getName(op.getLhs())
     << ", " << getName(op.getRhs()) << ");\n";
  return success();
}

LogicalResult CheddarEmitter::printOperation(MultOp op) {
  auto name = getName(op.getOutput());
  os << "Ct " << name << ";\n";
  os << getName(op.getCtx()) << "->Mult(" << name << ", "
     << getName(op.getLhs()) << ", " << getName(op.getRhs()) << ");\n";
  return success();
}

//===----------------------------------------------------------------------===//
// Ct-pt / ct-const ops
//===----------------------------------------------------------------------===//

LogicalResult CheddarEmitter::printOperation(AddPlainOp op) {
  auto name = getName(op.getOutput());
  os << "Ct " << name << ";\n";
  os << getName(op.getCtx()) << "->Add(" << name << ", "
     << getName(op.getCiphertext()) << ", " << getName(op.getPlaintext())
     << ");\n";
  return success();
}

LogicalResult CheddarEmitter::printOperation(SubPlainOp op) {
  auto name = getName(op.getOutput());
  os << "Ct " << name << ";\n";
  os << getName(op.getCtx()) << "->Sub(" << name << ", "
     << getName(op.getCiphertext()) << ", " << getName(op.getPlaintext())
     << ");\n";
  return success();
}

LogicalResult CheddarEmitter::printOperation(MultPlainOp op) {
  auto name = getName(op.getOutput());
  os << "Ct " << name << ";\n";
  os << getName(op.getCtx()) << "->Mult(" << name << ", "
     << getName(op.getCiphertext()) << ", " << getName(op.getPlaintext())
     << ");\n";
  return success();
}

LogicalResult CheddarEmitter::printOperation(AddConstOp op) {
  auto name = getName(op.getOutput());
  os << "Ct " << name << ";\n";
  os << getName(op.getCtx()) << "->Add(" << name << ", "
     << getName(op.getCiphertext()) << ", " << getName(op.getConstant())
     << ");\n";
  return success();
}

LogicalResult CheddarEmitter::printOperation(MultConstOp op) {
  auto name = getName(op.getOutput());
  os << "Ct " << name << ";\n";
  os << getName(op.getCtx()) << "->Mult(" << name << ", "
     << getName(op.getCiphertext()) << ", " << getName(op.getConstant())
     << ");\n";
  return success();
}

//===----------------------------------------------------------------------===//
// Unary ops
//===----------------------------------------------------------------------===//

LogicalResult CheddarEmitter::printOperation(NegOp op) {
  auto name = getName(op.getOutput());
  os << "Ct " << name << ";\n";
  os << getName(op.getCtx()) << "->Neg(" << name << ", "
     << getName(op.getInput()) << ");\n";
  return success();
}

LogicalResult CheddarEmitter::printOperation(RescaleOp op) {
  auto name = getName(op.getOutput());
  os << "Ct " << name << ";\n";
  os << getName(op.getCtx()) << "->Rescale(" << name << ", "
     << getName(op.getInput()) << ");\n";
  return success();
}

LogicalResult CheddarEmitter::printOperation(LevelDownOp op) {
  auto name = getName(op.getOutput());
  auto level = op.getTargetLevelAttr().getInt();
  os << "Ct " << name << ";\n";
  os << getName(op.getCtx()) << "->LevelDown(" << name << ", "
     << getName(op.getInput()) << ", " << level << ");\n";
  return success();
}

LogicalResult CheddarEmitter::printOperation(RelinearizeOp op) {
  auto name = getName(op.getOutput());
  os << "Ct " << name << ";\n";
  os << getName(op.getCtx()) << "->Relinearize(" << name << ", "
     << getName(op.getInput()) << ", " << getName(op.getMultKey()) << ");\n";
  return success();
}

LogicalResult CheddarEmitter::printOperation(RelinearizeRescaleOp op) {
  auto name = getName(op.getOutput());
  os << "Ct " << name << ";\n";
  os << getName(op.getCtx()) << "->RelinearizeRescale(" << name << ", "
     << getName(op.getInput()) << ", " << getName(op.getMultKey()) << ");\n";
  return success();
}

//===----------------------------------------------------------------------===//
// Fused compound ops
//===----------------------------------------------------------------------===//

LogicalResult CheddarEmitter::printOperation(HMultOp op) {
  auto name = getName(op.getOutput());
  bool rescale = op.getRescale();
  os << "Ct " << name << ";\n";
  os << getName(op.getCtx()) << "->HMult(" << name << ", "
     << getName(op.getLhs()) << ", " << getName(op.getRhs()) << ", "
     << getName(op.getMultKey()) << ", " << (rescale ? "true" : "false")
     << ");\n";
  return success();
}

LogicalResult CheddarEmitter::printOperation(HRotOp op) {
  auto name = getName(op.getOutput());
  os << "Ct " << name << ";\n";
  if (auto staticShift = op.getStaticShift()) {
    // Static shift: use the pre-fetched rotation key
    os << getName(op.getCtx()) << "->HRot(" << name << ", "
       << getName(op.getInput()) << ", " << getName(op.getRotKey()) << ", "
       << staticShift->getInt() << ");\n";
  } else {
    // Dynamic shift: look up the rotation key at runtime by the shift value.
    // The pre-fetched key (from GetRotKeyOp with distance=0) is a placeholder;
    // we need to get the actual key from the UserInterface at runtime.
    auto shiftName = getName(op.getDynamicShift());
    // Find the UserInterface from the GetRotKeyOp's operand
    auto getRotKeyOp = op.getRotKey().getDefiningOp<GetRotKeyOp>();
    if (getRotKeyOp) {
      os << getName(op.getCtx()) << "->HRot(" << name << ", "
         << getName(op.getInput()) << ", " << getName(getRotKeyOp.getUi())
         << ".GetRotationKey(" << shiftName << "), " << shiftName << ");\n";
    } else {
      // Fallback: use the key as-is (won't work correctly for dynamic shifts)
      os << getName(op.getCtx()) << "->HRot(" << name << ", "
         << getName(op.getInput()) << ", " << getName(op.getRotKey()) << ", "
         << shiftName << ");\n";
    }
  }
  return success();
}

LogicalResult CheddarEmitter::printOperation(HRotAddOp op) {
  auto name = getName(op.getOutput());
  auto dist = op.getDistanceAttr().getInt();
  os << "Ct " << name << ";\n";
  os << getName(op.getCtx()) << "->HRotAdd(" << name << ", "
     << getName(op.getInput()) << ", " << getName(op.getAddend()) << ", "
     << getName(op.getRotKey()) << ", " << dist << ");\n";
  return success();
}

LogicalResult CheddarEmitter::printOperation(HConjOp op) {
  auto name = getName(op.getOutput());
  os << "Ct " << name << ";\n";
  os << getName(op.getCtx()) << "->HConj(" << name << ", "
     << getName(op.getInput()) << ", " << getName(op.getConjKey()) << ");\n";
  return success();
}

LogicalResult CheddarEmitter::printOperation(HConjAddOp op) {
  auto name = getName(op.getOutput());
  os << "Ct " << name << ";\n";
  os << getName(op.getCtx()) << "->HConjAdd(" << name << ", "
     << getName(op.getInput()) << ", " << getName(op.getAddend()) << ", "
     << getName(op.getConjKey()) << ");\n";
  return success();
}

LogicalResult CheddarEmitter::printOperation(MadUnsafeOp op) {
  auto name = getName(op.getOutput());
  // MadUnsafe is in-place: res += a * const
  // We need to copy the accumulator first
  os << "Ct " << name << ";\n";
  os << getName(op.getCtx()) << "->Copy(" << name << ", "
     << getName(op.getAccumulator()) << ");\n";
  os << getName(op.getCtx()) << "->MadUnsafe(" << name << ", "
     << getName(op.getInput()) << ", " << getName(op.getConstant()) << ");\n";
  return success();
}

//===----------------------------------------------------------------------===//
// Extension ops
//===----------------------------------------------------------------------===//

LogicalResult CheddarEmitter::printOperation(BootOp op) {
  needsExtensionIncludes = true;
  auto name = getName(op.getOutput());
  os << "Ct " << name << ";\n";
  os << "std::dynamic_pointer_cast<BootContext<word>>(" << getName(op.getCtx())
     << ")->Boot(" << name << ", " << getName(op.getInput()) << ", "
     << getName(op.getEvkMap()) << ");\n";
  return success();
}

LogicalResult CheddarEmitter::printOperation(LinearTransformOp op) {
  needsExtensionIncludes = true;
  auto name = getName(op.getOutput());
  auto diagIndices = op.getDiagonalIndicesAttr().asArrayRef();
  auto level = op.getLevelAttr().getInt();
  auto logBSGS = op.getLogBabyStepGiantStepRatioAttr().getInt();
  auto diagonalsName = getName(op.getDiagonals());
  auto ctxName = getName(op.getCtx());
  auto evkMapName = getName(op.getEvkMap());
  auto inputName = getName(op.getInput());

  // Get diagonals tensor shape to determine slots
  auto diagType = cast<RankedTensorType>(op.getDiagonals().getType());
  int64_t numDiags = diagType.getShape()[0];
  int64_t slots = diagType.getShape()[1];

  // Compute baby-step / giant-step from logBSGS ratio
  // bs * gs >= numDiags, bs/gs ~ 2^logBSGS
  int64_t bs = 1;
  int64_t gs = numDiags;
  if (logBSGS > 0) {
    // Simple BSGS split: bs = ceil(sqrt(numDiags * 2^logBSGS))
    double ratio = std::pow(2.0, logBSGS);
    bs = static_cast<int64_t>(
        std::ceil(std::sqrt(static_cast<double>(numDiags) * ratio)));
    gs = (numDiags + bs - 1) / bs;
  }

  // Unique name for this linear transform
  std::string ltName = name + "_lt";
  std::string matName = name + "_mat";

  // Build the StripedMatrix from the diagonals tensor
  os << "StripedMatrix " << matName << "(" << numDiags << ", " << slots
     << ");\n";
  os << "{\n";
  os.indent();
  os << "// Populate diagonals from " << diagonalsName << "\n";
  os << "auto* data = " << diagonalsName << ".data();\n";
  for (int64_t i = 0; i < numDiags; ++i) {
    os << matName << "[" << diagIndices[i] << "] = std::vector<Complex>(data + "
       << i * slots << ", data + " << (i + 1) * slots << ");\n";
  }
  os.unindent();
  os << "}\n";

  // Construct LinearTransform
  os << "LinearTransform<word> " << ltName << "(" << ctxName << ", " << matName
     << ", " << level << ", static_cast<double>(1ULL << " << logDefaultScale
     << "), " << bs << ", " << gs << ");\n";

  // Evaluate
  os << "Ct " << name << ";\n";
  os << ltName << ".Evaluate(" << ctxName << ", " << name << ", " << inputName
     << ", " << evkMapName << ");\n";
  return success();
}

LogicalResult CheddarEmitter::printOperation(EvalPolyOp op) {
  needsExtensionIncludes = true;
  auto name = getName(op.getOutput());
  auto ctxName = getName(op.getCtx());
  auto evkMapName = getName(op.getEvkMap());
  auto inputName = getName(op.getInput());
  auto coeffsAttr = op.getCoefficientsAttr();

  // Emit coefficient vector
  std::string coeffsName = name + "_coeffs";
  os << "std::vector<double> " << coeffsName << " = {";
  for (unsigned i = 0; i < coeffsAttr.size(); ++i) {
    if (i > 0) os << ", ";
    auto floatAttr = cast<FloatAttr>(coeffsAttr[i]);
    SmallString<16> str;
    floatAttr.getValue().toString(str, /*FormatPrecision=*/15);
    os << str;
  }
  os << "};\n";

  // Construct EvalPoly and evaluate
  std::string evalPolyName = name + "_ep";
  os << "EvalPoly<word> " << evalPolyName << "(" << ctxName << ", "
     << evkMapName << ");\n";
  os << "Ct " << name << ";\n";
  os << evalPolyName << ".Evaluate(" << ctxName << ", " << name << ", "
     << inputName << ", " << coeffsName << ");\n";
  return success();
}

//===----------------------------------------------------------------------===//
// SCF control flow ops
//===----------------------------------------------------------------------===//

LogicalResult CheddarEmitter::printOperation(scf::ForOp op) {
  // Initialize iter args
  for (unsigned i = 0; i < op.getNumRegionIterArgs(); ++i) {
    Value result = op.getResults()[i];
    Value init = op.getInitArgs()[i];
    Value iterArg = op.getRegionIterArgs()[i];
    if (variableNames->contains(result) && variableNames->contains(init) &&
        getName(result) == getName(init))
      continue;
    auto typeStr = convertType(iterArg.getType());
    if (failed(typeStr))
      return op.emitOpError("failed to convert iter arg type");
    os << *typeStr << " " << getName(iterArg) << " = std::move("
       << getName(init) << ");\n";
  }

  // for (int64_t iv = lb; iv < ub; iv += step)
  auto getVal = [&](Value v) -> std::string {
    if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
        return std::to_string(intAttr.getInt());
    }
    return getName(v);
  };
  os << "for (int64_t " << getName(op.getInductionVar()) << " = "
     << getVal(op.getLowerBound()) << "; " << getName(op.getInductionVar())
     << " < " << getVal(op.getUpperBound()) << "; "
     << getName(op.getInductionVar()) << " += " << getVal(op.getStep())
     << ") {\n";
  os.indent();
  for (Operation &bodyOp : *op.getBody()) {
    if (failed(translate(bodyOp))) return failure();
  }
  os.unindent();
  os << "}\n";

  // Forward iter args to results
  for (unsigned i = 0; i < op.getNumResults(); ++i) {
    Value opResult = op.getResult(i);
    Value iterArg = op.getRegionIterArg(i);
    if (getName(opResult) != getName(iterArg)) {
      auto typeStr = convertType(opResult.getType());
      if (failed(typeStr)) return failure();
      os << *typeStr << " " << getName(opResult) << " = std::move("
         << getName(iterArg) << ");\n";
    }
  }
  return success();
}

LogicalResult CheddarEmitter::printOperation(scf::IfOp op) {
  // Declare result variables
  for (unsigned i = 0; i < op.getNumResults(); ++i) {
    auto typeStr = convertType(op.getResults()[i].getType());
    if (failed(typeStr))
      return op.emitOpError("failed to convert if result type");
    os << *typeStr << " " << getName(op.getResults()[i]) << ";\n";
  }

  os << "if (" << getName(op.getCondition()) << ") {\n";
  os.indent();
  for (Operation &thenOp : *op.thenBlock()) {
    if (failed(translate(thenOp))) return failure();
  }
  os.unindent();
  os << "} else {\n";
  os.indent();
  for (Operation &elseOp : *op.elseBlock()) {
    if (failed(translate(elseOp))) return failure();
  }
  os.unindent();
  os << "}\n";
  return success();
}

LogicalResult CheddarEmitter::printOperation(scf::YieldOp op) {
  ValueRange destValues =
      llvm::TypeSwitch<Operation *, ValueRange>(op->getParentOp())
          .Case<scf::ForOp>(
              [&](auto forOp) { return forOp.getRegionIterArgs(); })
          .Case<scf::IfOp>([&](auto ifOp) { return ifOp.getResults(); })
          .Default([&](auto) { return ValueRange{}; });

  for (unsigned i = 0; i < op.getNumOperands(); ++i) {
    os << getName(destValues[i]) << " = std::move(" << getName(op.getOperand(i))
       << ");\n";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Tensor ops (tensors are flattened to std::vector)
//===----------------------------------------------------------------------===//

LogicalResult CheddarEmitter::printOperation(tensor::EmptyOp op) {
  auto resultType = op.getResult().getType();
  auto elemType = resultType.getElementType();
  auto typeStr = convertType(elemType);
  if (failed(typeStr)) return op.emitOpError("failed to convert element type");
  bool isMoveOnly = isa<CiphertextType, PlaintextType, ConstantType>(elemType);
  if (isMoveOnly) {
    // Move-only: reserve without constructing (will be filled via insert)
    os << "std::vector<" << *typeStr << "> " << getName(op.getResult())
       << ";\n";
    os << getName(op.getResult()) << ".resize(" << resultType.getNumElements()
       << ");\n";
  } else {
    os << "std::vector<" << *typeStr << "> " << getName(op.getResult()) << "("
       << resultType.getNumElements() << ");\n";
  }
  return success();
}

LogicalResult CheddarEmitter::printOperation(tensor::ExtractOp op) {
  auto resultType = op.getResult().getType();
  bool isMoveOnly =
      isa<CiphertextType, PlaintextType, ConstantType>(resultType);
  if (isMoveOnly) {
    // Reference to avoid copying move-only types
    os << "auto& " << getName(op.getResult()) << " = "
       << getName(op.getTensor()) << "[";
  } else {
    auto typeStr = convertType(resultType);
    if (failed(typeStr)) return failure();
    os << *typeStr << " " << getName(op.getResult()) << " = "
       << getName(op.getTensor()) << "[";
  }
  os << flattenIndexExpression(op.getTensor().getType(), op.getIndices(),
                               [&](Value value) { return getName(value); });
  os << "];\n";
  return success();
}

LogicalResult CheddarEmitter::printOperation(tensor::InsertOp op) {
  bool inPlace = variableNames->contains(op.getResult()) &&
                 variableNames->contains(op.getDest()) &&
                 getName(op.getResult()) == getName(op.getDest());
  std::string resultName;
  if (inPlace) {
    resultName = getName(op.getResult());
  } else {
    // Move dest (important for move-only types like Ct)
    auto typeStr = convertType(
        cast<RankedTensorType>(op.getResult().getType()).getElementType());
    if (failed(typeStr)) return failure();
    resultName = getName(op.getResult());
    os << "auto " << resultName << " = std::move(" << getName(op.getDest())
       << ");\n";
  }
  os << resultName << "[";
  os << flattenIndexExpression(op.getResult().getType(), op.getIndices(),
                               [&](Value value) { return getName(value); });
  os << "] = std::move(" << getName(op.getScalar()) << ");\n";
  return success();
}

LogicalResult CheddarEmitter::printOperation(tensor::FromElementsOp op) {
  auto elemType = getElementTypeOrSelf(op.getResult().getType());
  auto typeStr = convertType(elemType);
  if (failed(typeStr)) return failure();
  bool isMoveOnly = isa<CiphertextType, PlaintextType, ConstantType>(elemType);
  if (isMoveOnly) {
    // Move-only types: reserve + emplace_back with std::move
    os << "std::vector<" << *typeStr << "> " << getName(op.getResult())
       << ";\n";
    os << getName(op.getResult()) << ".reserve(" << op.getNumOperands()
       << ");\n";
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      os << getName(op.getResult()) << ".emplace_back(std::move("
         << getName(op.getOperand(i)) << "));\n";
    }
  } else {
    os << "std::vector<" << *typeStr << "> " << getName(op.getResult())
       << " = {";
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      if (i > 0) os << ", ";
      os << getName(op.getOperand(i));
    }
    os << "};\n";
  }
  return success();
}

LogicalResult CheddarEmitter::printOperation(tensor::ExpandShapeOp op) {
  // Tensors are flat — expand_shape is a no-op alias.
  SliceVerificationResult res =
      isRankReducedType(op.getResultType(), op.getSrcType());
  if (res != SliceVerificationResult::Success) {
    return op.emitError()
           << "Only rank-reduced types are supported for ExpandShapeOp";
  }
  variableNames->mapValueNameToValue(op.getResult(), op.getSrc());
  return success();
}

LogicalResult CheddarEmitter::printOperation(tensor::ExtractSliceOp op) {
  auto resultType = op.getResultType();
  auto typeStr = convertType(resultType.getElementType());
  if (failed(typeStr)) return failure();

  auto getOffsetStr = [&](OpFoldResult ofr) -> std::string {
    if (auto attr = dyn_cast<Attribute>(ofr))
      return std::to_string(cast<IntegerAttr>(attr).getInt());
    return getName(cast<Value>(ofr));
  };

  // Simple contiguous case: just take a sub-span
  auto offsets = op.getMixedOffsets();
  auto sizes = op.getMixedSizes();
  int64_t numElements = resultType.getNumElements();

  os << "std::vector<" << *typeStr << "> " << getName(op.getResult()) << "("
     << getName(op.getSource()) << ".begin() + " << getOffsetStr(offsets[0]);
  // Flatten multi-dim offset
  auto srcType = op.getSourceType();
  for (unsigned i = 1; i < offsets.size(); ++i) {
    int64_t stride = 1;
    for (unsigned j = i + 1; j < srcType.getRank(); ++j)
      stride *= srcType.getDimSize(j);
    os << " + " << getOffsetStr(offsets[i]) << " * " << stride;
  }
  os << ", " << getName(op.getSource()) << ".begin() + "
     << getOffsetStr(offsets[0]);
  for (unsigned i = 1; i < offsets.size(); ++i) {
    int64_t stride = 1;
    for (unsigned j = i + 1; j < srcType.getRank(); ++j)
      stride *= srcType.getDimSize(j);
    os << " + " << getOffsetStr(offsets[i]) << " * " << stride;
  }
  os << " + " << numElements << ");\n";
  return success();
}

LogicalResult CheddarEmitter::printOperation(tensor::InsertSliceOp op) {
  bool inPlace = variableNames->contains(op.getResult()) &&
                 variableNames->contains(op.getDest()) &&
                 getName(op.getResult()) == getName(op.getDest());
  std::string resultName;
  if (inPlace) {
    resultName = getName(op.getResult());
  } else {
    auto typeStr = convertType(
        cast<RankedTensorType>(op.getResult().getType()).getElementType());
    if (failed(typeStr)) return failure();
    resultName = getName(op.getResult());
    os << "auto " << resultName << " = std::move(" << getName(op.getDest())
       << ");\n";
  }
  // Copy source into dest at offset
  auto offsets = op.getMixedOffsets();
  auto getOffsetStr = [&](OpFoldResult ofr) -> std::string {
    if (auto attr = dyn_cast<Attribute>(ofr))
      return std::to_string(cast<IntegerAttr>(attr).getInt());
    return getName(cast<Value>(ofr));
  };
  os << "std::copy(" << getName(op.getSource()) << ".begin(), "
     << getName(op.getSource()) << ".end(), " << resultName << ".begin() + "
     << getOffsetStr(offsets[0]);
  auto destType = op.getType();
  for (unsigned i = 1; i < offsets.size(); ++i) {
    int64_t stride = 1;
    for (unsigned j = i + 1; j < destType.getRank(); ++j)
      stride *= destType.getDimSize(j);
    os << " + " << getOffsetStr(offsets[i]) << " * " << stride;
  }
  os << ");\n";
  return success();
}

//===----------------------------------------------------------------------===//
// Additional arith ops
//===----------------------------------------------------------------------===//

LogicalResult CheddarEmitter::printOperation(arith::AddIOp op) {
  auto typeStr = convertType(op.getResult().getType());
  if (failed(typeStr)) return failure();
  os << *typeStr << " " << getName(op.getResult()) << " = "
     << getName(op.getLhs()) << " + " << getName(op.getRhs()) << ";\n";
  return success();
}

LogicalResult CheddarEmitter::printOperation(arith::SubIOp op) {
  auto typeStr = convertType(op.getResult().getType());
  if (failed(typeStr)) return failure();
  os << *typeStr << " " << getName(op.getResult()) << " = "
     << getName(op.getLhs()) << " - " << getName(op.getRhs()) << ";\n";
  return success();
}

LogicalResult CheddarEmitter::printOperation(arith::RemSIOp op) {
  auto typeStr = convertType(op.getResult().getType());
  if (failed(typeStr)) return failure();
  os << *typeStr << " " << getName(op.getResult()) << " = "
     << getName(op.getLhs()) << " % " << getName(op.getRhs()) << ";\n";
  return success();
}

LogicalResult CheddarEmitter::printOperation(arith::CmpIOp op) {
  std::string cmpOp;
  switch (op.getPredicate()) {
    case arith::CmpIPredicate::eq:
      cmpOp = "==";
      break;
    case arith::CmpIPredicate::ne:
      cmpOp = "!=";
      break;
    case arith::CmpIPredicate::slt:
    case arith::CmpIPredicate::ult:
      cmpOp = "<";
      break;
    case arith::CmpIPredicate::sle:
    case arith::CmpIPredicate::ule:
      cmpOp = "<=";
      break;
    case arith::CmpIPredicate::sgt:
    case arith::CmpIPredicate::ugt:
      cmpOp = ">";
      break;
    case arith::CmpIPredicate::sge:
    case arith::CmpIPredicate::uge:
      cmpOp = ">=";
      break;
  }
  os << "bool " << getName(op.getResult()) << " = " << getName(op.getLhs())
     << " " << cmpOp << " " << getName(op.getRhs()) << ";\n";
  return success();
}

LogicalResult CheddarEmitter::printOperation(arith::IndexCastOp op) {
  auto typeStr = convertType(op.getOut().getType());
  if (failed(typeStr)) return failure();
  os << *typeStr << " " << getName(op.getOut()) << " = static_cast<" << *typeStr
     << ">(" << getName(op.getIn()) << ");\n";
  return success();
}

//===----------------------------------------------------------------------===//
// Translation registration
//===----------------------------------------------------------------------===//

LogicalResult translateToCheddar(Operation *op, llvm::raw_ostream &os,
                                 bool use64Bit, const std::string &paramsJson) {
  SelectVariableNames variableNames(op);
  // Two-pass: buffer body, then emit prelude + body
  std::string bufferedStr;
  llvm::raw_string_ostream strOs(bufferedStr);
  CheddarEmitter emitter(strOs, &variableNames, use64Bit, paramsJson);
  LogicalResult result = emitter.translate(*op);
  if (failed(result)) return result;

  emitter.emitPrelude(os);
  os << strOs.str();
  return success();
}

struct CheddarTranslateOptions {
  llvm::cl::opt<bool> use64Bit{"cheddar-use-64bit",
                               llvm::cl::desc("Use uint64_t word type"),
                               llvm::cl::init(true)};
  llvm::cl::opt<std::string> paramsJson{
      "cheddar-params-json",
      llvm::cl::desc("Path to CHEDDAR parameter JSON file"),
      llvm::cl::init("")};
};
static llvm::ManagedStatic<CheddarTranslateOptions> cheddarTranslateOptions;

void registerCheddarTranslateOptions() { *cheddarTranslateOptions; }

static void registerRelevantDialects(DialectRegistry &registry) {
  registry.insert<affine::AffineDialect, arith::ArithDialect, func::FuncDialect,
                  scf::SCFDialect, tensor::TensorDialect,
                  tensor_ext::TensorExtDialect, cheddar::CheddarDialect>();
}

void registerToCheddarTranslation() {
  TranslateFromMLIRRegistration reg(
      "emit-cheddar",
      "translate the cheddar dialect to C++ code against the CHEDDAR API",
      [](Operation *op, llvm::raw_ostream &output) {
        return translateToCheddar(op, output, cheddarTranslateOptions->use64Bit,
                                  cheddarTranslateOptions->paramsJson);
      },
      registerRelevantDialects);
}

LogicalResult translateToCheddarHeader(Operation *op, llvm::raw_ostream &os,
                                       bool use64Bit) {
  // Emit a proper C++ header: includes, type aliases, function declarations.
  auto moduleOp = dyn_cast<ModuleOp>(op);
  if (!moduleOp) return failure();

  SelectVariableNames variableNames(op);

  os << "#pragma once\n";
  os << kStdIncludes;
  os << kCheddarInclude;

  // Check if extensions are needed
  bool needsExtension = false;
  moduleOp->walk([&](Operation *innerOp) {
    if (isa<LinearTransformOp, EvalPolyOp, BootOp>(innerOp))
      needsExtension = true;
  });
  if (needsExtension) os << kCheddarExtensionInclude;

  os << "\n";
  if (use64Bit) {
    os << kTypeAliasPrelude64;
  } else {
    os << kTypeAliasPrelude32;
  }
  os << "\n";

  // Emit function declarations
  for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
    auto funcType = funcOp.getFunctionType();
    auto resultTypes = funcType.getResults();
    auto argTypes = funcType.getInputs();

    // Two-pass: buffer to determine return type string
    CheddarEmitter tempEmitter(llvm::nulls(), &variableNames, use64Bit);

    // Return type
    if (resultTypes.empty()) {
      os << "void ";
    } else if (resultTypes.size() == 1) {
      auto typeStr = tempEmitter.convertType(resultTypes[0]);
      if (failed(typeStr)) return failure();
      os << *typeStr << " ";
    } else {
      os << "std::tuple<";
      for (unsigned i = 0; i < resultTypes.size(); ++i) {
        if (i > 0) os << ", ";
        auto typeStr = tempEmitter.convertType(resultTypes[i]);
        if (failed(typeStr)) return failure();
        os << *typeStr;
      }
      os << "> ";
    }

    // Function name and arg types
    os << funcOp.getName() << "(";
    for (unsigned i = 0; i < argTypes.size(); ++i) {
      if (i > 0) os << ", ";
      auto typeStr = tempEmitter.convertType(argTypes[i], /*asArg=*/true);
      if (failed(typeStr)) return failure();
      os << *typeStr;
    }
    os << ");\n";
  }

  return success();
}

void registerToCheddarHeaderTranslation() {
  TranslateFromMLIRRegistration reg(
      "emit-cheddar-header",
      "translate the cheddar dialect to a C++ header file",
      [](Operation *op, llvm::raw_ostream &output) {
        return translateToCheddarHeader(op, output,
                                        cheddarTranslateOptions->use64Bit);
      },
      registerRelevantDialects);
}

}  // namespace cheddar
}  // namespace heir
}  // namespace mlir
