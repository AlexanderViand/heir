#include "lib/Target/Cheddar/CheddarEmitter.h"

#include <cmath>
#include <string>

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/Cheddar/IR/CheddarDialect.h"
#include "lib/Dialect/Cheddar/IR/CheddarOps.h"
#include "lib/Dialect/Cheddar/IR/CheddarTypes.h"
#include "lib/Target/Cheddar/CheddarTemplates.h"
#include "lib/Utils/TargetUtils.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "llvm/include/llvm/Support/ManagedStatic.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
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

FailureOr<std::string> CheddarEmitter::convertType(Type type) {
  return llvm::TypeSwitch<Type, FailureOr<std::string>>(type)
      .Case<ContextType>([](auto) { return std::string("CtxPtr"); })
      .Case<ParameterType>([](auto) { return std::string("Param"); })
      .Case<EncoderType>([](auto) { return std::string("Enc&"); })
      .Case<UserInterfaceType>([](auto) { return std::string("UI"); })
      .Case<CiphertextType>([](auto) { return std::string("Ct"); })
      .Case<PlaintextType>([](auto) { return std::string("Pt"); })
      .Case<ConstantType>([](auto) { return std::string("Const"); })
      .Case<EvalKeyType>([](auto) { return std::string("const Evk&"); })
      .Case<EvkMapType>([](auto) { return std::string("const EvkMapT&"); })
      .Case<RankedTensorType>(
          [](RankedTensorType type) -> FailureOr<std::string> {
            auto elemType = type.getElementType();
            if (elemType.isF64() || elemType.isF32()) {
              return std::string("std::vector<Complex>");
            }
            return failure();
          })
      .Case<FloatType>([](auto) { return std::string("double"); })
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
          .Default([&](Operation &) {
            return op.emitOpError("unable to find printer for op");
          });
  return status;
}

//===----------------------------------------------------------------------===//
// Module / Function
//===----------------------------------------------------------------------===//

LogicalResult CheddarEmitter::printOperation(ModuleOp moduleOp) {
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
    auto typeStr = convertType(argTypes[i]);
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
    // Emit as std::vector<Complex>
    auto name = getName(result);
    os << "std::vector<Complex> " << name << " = {";
    bool first = true;
    for (auto val : denseAttr.getValues<APFloat>()) {
      if (!first) os << ", ";
      first = false;
      SmallString<16> str;
      val.toString(str, /*FormatPrecision=*/15);
      os << "Complex(" << str << ", 0.0)";
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
  os << getName(op.getUi()) << ".PrepareRotationKey(" << dist << ");\n";
  return success();
}

//===----------------------------------------------------------------------===//
// Encode / Encrypt / Decrypt
//===----------------------------------------------------------------------===//

LogicalResult CheddarEmitter::printOperation(EncodeOp op) {
  auto level = op.getLevelAttr().getInt();
  auto scale = op.getScaleAttr().getInt();
  auto name = getName(op.getPlaintext());
  os << "Pt " << name << ";\n";
  os << getName(op.getEncoder()) << ".Encode(" << name << ", " << level << ", "
     << scale << ", " << getName(op.getMessage()) << ");\n";
  return success();
}

LogicalResult CheddarEmitter::printOperation(EncodeConstantOp op) {
  auto level = op.getLevelAttr().getInt();
  auto scale = op.getScaleAttr().getInt();
  auto name = getName(op.getConstant());
  os << "Const " << name << ";\n";
  os << getName(op.getEncoder()) << ".EncodeConstant(" << name << ", " << level
     << ", " << scale << ", " << getName(op.getValue()) << ");\n";
  return success();
}

LogicalResult CheddarEmitter::printOperation(DecodeOp op) {
  auto name = getName(op.getMessage());
  os << "std::vector<Complex> " << name << ";\n";
  os << getName(op.getEncoder()) << ".Decode(" << name << ", "
     << getName(op.getPlaintext()) << ");\n";
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
  auto dist = op.getDistanceAttr().getInt();
  os << "Ct " << name << ";\n";
  os << getName(op.getCtx()) << "->HRot(" << name << ", "
     << getName(op.getInput()) << ", " << getName(op.getRotKey()) << ", "
     << dist << ");\n";
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
  // TODO: derive scale from scheme params rather than hardcoding
  os << "LinearTransform<word> " << ltName << "(" << ctxName << ", " << matName
     << ", " << level << ", static_cast<double>(1ULL << "
     << 40  // logDefaultScale placeholder
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
  // TODO: Emit proper EvalPoly code
  os << "// EvalPoly: coefficients=" << op.getCoefficientsAttr() << "\n";
  os << "Ct " << name << "; // TODO: implement EvalPoly emission\n";
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
  registry.insert<arith::ArithDialect, func::FuncDialect, tensor::TensorDialect,
                  cheddar::CheddarDialect>();
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
      auto typeStr = tempEmitter.convertType(argTypes[i]);
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
