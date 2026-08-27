#include <cctype>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>

#include "lib/Conversions/CheddarToEmitC/CheddarToEmitC.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Utils/EntryInterfaceUtils.h"
#include "llvm/include/llvm/ADT/DenseSet.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"          // from @llvm-project
#include "llvm/include/llvm/ADT/StringExtras.h"         // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"            // from @llvm-project
#include "mlir/include/mlir/Dialect/EmitC/IR/EmitC.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"              // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/TypeRange.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project

namespace mlir::heir {

#define GEN_PASS_DEF_CHEDDAREMITCENTRYINTERFACE
#include "lib/Conversions/CheddarToEmitC/CheddarToEmitC.h.inc"

namespace {

using emitc::CallOpaqueOp;
using emitc::ClassOp;
using emitc::FieldOp;
using emitc::FileOp;
using emitc::FuncOp;
using emitc::IncludeOp;
using emitc::LValueType;
using emitc::MemberCallOpaqueOp;
using emitc::MemberOp;
using emitc::OpaqueAttr;
using emitc::OpaqueType;
using emitc::PointerType;
using emitc::ReturnOp;
using emitc::VariableOp;
using emitc::VerbatimOp;

std::string sanitizeIdentifier(StringRef value) {
  std::string result;
  result.reserve(value.size());
  for (char c : value)
    result.push_back(std::isalnum(static_cast<unsigned char>(c)) ? c : '_');
  if (result.empty() || std::isdigit(static_cast<unsigned char>(result[0])))
    result.insert(result.begin(), '_');
  return result;
}

StringRef opaqueName(Type type) {
  if (auto opaque = dyn_cast<OpaqueType>(type)) return opaque.getValue();
  return {};
}

std::string trimReference(StringRef name) {
  name = name.trim();
  if (name.consume_front("const ")) name = name.trim();
  if (name.consume_back("&")) name = name.rtrim();
  return name.str();
}

std::string cppTypeName(Type type) {
  if (auto opaque = dyn_cast<OpaqueType>(type)) return opaque.getValue().str();
  if (auto pointer = dyn_cast<PointerType>(type))
    return cppTypeName(pointer.getPointee()) + "*";
  if (type.isF16()) return "_Float16";
  if (type.isF32()) return "float";
  if (type.isF64()) return "double";
  if (type.isIndex()) return "std::size_t";
  if (auto integer = dyn_cast<IntegerType>(type)) {
    if (integer.getWidth() == 1) return "bool";
    std::string prefix = integer.isUnsigned() ? "std::uint" : "std::int";
    return prefix + std::to_string(integer.getWidth()) + "_t";
  }
  return {};
}

std::string stdArrayName(ArrayRef<int64_t> shape, StringRef element) {
  std::string result = element.str();
  for (int64_t dimension : llvm::reverse(shape))
    result = "std::array<" + result + ", " + std::to_string(dimension) + ">";
  return result;
}

FailureOr<std::string> logicalCppType(Type type, Operation* diagnostic) {
  if (auto tensor = dyn_cast<RankedTensorType>(type)) {
    if (!tensor.hasStaticShape())
      return diagnostic->emitError(
          "entry interface requires statically shaped logical tensors");
    std::string element = cppTypeName(tensor.getElementType());
    if (element.empty())
      return diagnostic->emitError()
             << "unsupported logical tensor element type "
             << tensor.getElementType();
    if (tensor.getRank() == 0) return element;
    return stdArrayName(tensor.getShape(), element);
  }
  std::string result = cppTypeName(type);
  if (result.empty())
    return diagnostic->emitError() << "unsupported logical type " << type;
  return result;
}

SmallVector<unsigned> getDestinationArguments(func::FuncOp function) {
  SmallVector<unsigned> result;
  for (unsigned i = 0; i < function.getNumArguments(); ++i)
    if (function.getArgAttr(i, "bufferize.result")) result.push_back(i);
  return result;
}

Type storedType(Type argumentType) {
  if (auto opaque = dyn_cast<OpaqueType>(argumentType))
    return OpaqueType::get(argumentType.getContext(),
                           trimReference(opaque.getValue()));
  return argumentType;
}

SmallVector<Type> getDestinationTypes(func::FuncOp function) {
  SmallVector<Type> result;
  for (unsigned index : getDestinationArguments(function))
    result.push_back(storedType(function.getArgumentTypes()[index]));
  return result;
}

std::string tupleTypeName(ArrayRef<Type> fields) {
  std::string result = "std::tuple<";
  for (auto [index, type] : llvm::enumerate(fields)) {
    if (index) result += ", ";
    result += cppTypeName(type);
  }
  return result + ">";
}

bool isContextPointer(Type type) {
  auto pointer = dyn_cast<PointerType>(type);
  if (!pointer) return false;
  StringRef pointee = opaqueName(pointer.getPointee());
  return pointee == "Context<word>" || pointee == "BootContext<word>";
}

bool isUserInterfacePointer(Type type) {
  auto pointer = dyn_cast<PointerType>(type);
  return pointer && opaqueName(pointer.getPointee()) == "UserInterface<word>";
}

bool isSupportArgument(Type type) {
  if (isContextPointer(type) || isUserInterfacePointer(type)) return true;
  StringRef name = opaqueName(type);
  return name.contains("Encoder<word>") ||
         name.contains("EvaluationKey<word>") || name.contains("EvkMap<word>");
}

std::string contextTypeName(const EntryFunctions& functions) {
  bool hasContext = false;
  SmallVector<func::FuncOp> candidates;
  for (const auto& helper : functions.inputHelpers)
    candidates.push_back(helper.second);
  candidates.push_back(functions.preprocess);
  candidates.push_back(functions.evaluate);
  for (const auto& helper : functions.outputHelpers)
    candidates.push_back(helper.second);
  for (func::FuncOp function : candidates) {
    if (!function) continue;
    for (Type type : function.getArgumentTypes()) {
      if (!isContextPointer(type)) continue;
      StringRef name = opaqueName(cast<PointerType>(type).getPointee());
      if (name == "BootContext<word>") return name.str();
      hasContext = true;
    }
  }
  return hasContext ? "Context<word>" : "";
}

void emitVerbatim(OpBuilder& builder, Location loc, StringRef text) {
  VerbatimOp::create(builder, loc, text, ValueRange{});
}

void emitInclude(OpBuilder& builder, Location loc, StringRef include,
                 bool standard = true) {
  IncludeOp::create(builder, loc, include, standard);
}

FuncOp createEmitCFunction(OpBuilder& builder, Location loc, StringRef name,
                           TypeRange inputs, TypeRange results,
                           bool declaration) {
  auto type = FunctionType::get(builder.getContext(), inputs, results);
  auto function = FuncOp::create(builder, loc, name, type);
  if (!declaration) function.addEntryBlock();
  return function;
}

Value createLocal(OpBuilder& builder, Location loc, StringRef typeName) {
  Type type = OpaqueType::get(builder.getContext(), typeName);
  return VariableOp::create(builder, loc, LValueType::get(type),
                            OpaqueAttr::get(builder.getContext(), ""));
}

Value moveValue(OpBuilder& builder, Location loc, Value value,
                StringRef resultType) {
  return CallOpaqueOp::create(
             builder, loc,
             TypeRange{OpaqueType::get(builder.getContext(), resultType)},
             "std::move", value)
      .getResult(0);
}

Value getTupleElement(OpBuilder& builder, Location loc, Value tuple,
                      unsigned index, Type fieldType, bool isConst = false) {
  std::string resultType = cppTypeName(fieldType) + "&";
  if (isConst) resultType = "const " + resultType;
  return CallOpaqueOp::create(
             builder, loc,
             TypeRange{OpaqueType::get(builder.getContext(), resultType)},
             "std::get<" + std::to_string(index) + ">", tuple)
      .getResult(0);
}

Value getInputData(OpBuilder& builder, Location loc, Value input,
                   Type expectedType) {
  Type pointerType = expectedType;
  if (auto arrayType = dyn_cast<emitc::ArrayType>(expectedType)) {
    Type pointeeType = arrayType.getElementType();
    if (arrayType.getShape().size() > 1)
      pointeeType = emitc::ArrayType::get(arrayType.getShape().drop_front(),
                                          arrayType.getElementType());
    pointerType = PointerType::get(pointeeType);
  } else if (!isa<PointerType>(expectedType)) {
    return input;
  }
  return CallOpaqueOp::create(builder, loc, TypeRange{pointerType},
                              "heir::data", input)
      .getResult(0);
}

struct SupportValues {
  Value contextPointer;
  Value encoder;
  Value userInterface;
  Value evaluationKey;
  Value evaluationKeyMap;
};

SupportValues buildSupportValues(OpBuilder& builder, Location loc,
                                 Value context, Value key) {
  auto* ctx = builder.getContext();
  SupportValues values;
  Type contextPointer = PointerType::get(OpaqueType::get(ctx, "Context"));
  values.contextPointer =
      CallOpaqueOp::create(builder, loc, TypeRange{contextPointer},
                           "std::addressof", context)
          .getResult(0);
  values.encoder =
      CallOpaqueOp::create(
          builder, loc, TypeRange{OpaqueType::get(ctx, "const Encoder<word>&")},
          "heir::getEncoder", context)
          .getResult(0);
  Type uiPointer =
      PointerType::get(OpaqueType::get(ctx, "UserInterface<word>"));
  values.userInterface =
      CallOpaqueOp::create(builder, loc, TypeRange{uiPointer},
                           "static_cast<UserInterface<word>*>", key)
          .getResult(0);
  values.evaluationKey =
      MemberCallOpaqueOp::create(
          builder, loc,
          TypeRange{OpaqueType::get(ctx, "const EvaluationKey<word>&")},
          values.userInterface, "GetMultiplicationKey", ArrayAttr{},
          ArrayAttr{}, ValueRange{})
          .getResult(0);
  values.evaluationKeyMap =
      MemberCallOpaqueOp::create(
          builder, loc, TypeRange{OpaqueType::get(ctx, "const EvkMap<word>&")},
          values.userInterface, "GetEvkMap", ArrayAttr{}, ArrayAttr{},
          ValueRange{})
          .getResult(0);
  return values;
}

Value getSupportValue(Type expectedType, const SupportValues& values) {
  if (isContextPointer(expectedType)) return values.contextPointer;
  if (isUserInterfacePointer(expectedType)) return values.userInterface;
  StringRef name = opaqueName(expectedType);
  if (name.contains("Encoder<word>")) return values.encoder;
  if (name.contains("EvaluationKey<word>")) return values.evaluationKey;
  if (name.contains("EvkMap<word>")) return values.evaluationKeyMap;
  return {};
}

CallOpaqueOp callInternal(OpBuilder& builder, Location loc,
                          func::FuncOp function, ValueRange arguments) {
  return CallOpaqueOp::create(
      builder, loc, function.getResultTypes(),
      "::heir::generated::detail::" + function.getSymName().str(), arguments);
}

SmallVector<Type> getDataArgumentTypes(func::FuncOp function) {
  SmallVector<Type> result;
  for (auto [index, type] : llvm::enumerate(function.getArgumentTypes())) {
    if (function.getArgAttr(index, "bufferize.result") ||
        isSupportArgument(type))
      continue;
    result.push_back(storedType(type));
  }
  return result;
}

LogicalResult addResourceDirectoryArguments(func::FuncOp root) {
  ModuleOp module = root->getParentOfType<ModuleOp>();
  auto lookupCallee = [&](StringRef callee) {
    return module.lookupSymbol<func::FuncOp>(callee);
  };
  llvm::DenseSet<Operation*> reachableSet;
  SmallVector<func::FuncOp> reachable;
  std::function<void(func::FuncOp)> visit = [&](func::FuncOp function) {
    if (!reachableSet.insert(function.getOperation()).second) return;
    reachable.push_back(function);
    function.walk([&](func::CallOp call) {
      if (auto callee = lookupCallee(call.getCallee())) visit(callee);
    });
    function.walk([&](CallOpaqueOp call) {
      if (auto callee = lookupCallee(call.getCallee())) visit(callee);
    });
  };
  visit(root);

  llvm::DenseSet<Operation*> needsDirectory;
  for (func::FuncOp function : reachable) {
    function.walk([&](CallOpaqueOp call) {
      if (call.getCallee() == "heir::loadResource")
        needsDirectory.insert(function.getOperation());
    });
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (func::FuncOp function : reachable) {
      if (needsDirectory.contains(function.getOperation())) continue;
      function.walk([&](func::CallOp call) {
        if (auto callee = lookupCallee(call.getCallee());
            callee && needsDirectory.contains(callee.getOperation())) {
          changed |= needsDirectory.insert(function.getOperation()).second;
        }
      });
      function.walk([&](CallOpaqueOp call) {
        if (auto callee = lookupCallee(call.getCallee());
            callee && needsDirectory.contains(callee.getOperation())) {
          changed |= needsDirectory.insert(function.getOperation()).second;
        }
      });
    }
  }
  if (!needsDirectory.contains(root.getOperation())) return success();

  auto* ctx = root.getContext();
  Type directoryType = OpaqueType::get(ctx, "std::string_view");
  for (func::FuncOp function : reachable) {
    if (!needsDirectory.contains(function.getOperation())) continue;
    unsigned directoryIndex = function.getNumArguments();
    if (failed(function.insertArgument(directoryIndex, directoryType,
                                       DictionaryAttr{}, function.getLoc())))
      return function.emitOpError("failed to add resource directory argument");

    SmallVector<CallOpaqueOp> resourceLoads;
    function.walk([&](CallOpaqueOp call) {
      if (call.getCallee() == "heir::loadResource")
        resourceLoads.push_back(call);
    });
    for (CallOpaqueOp call : resourceLoads) {
      OpBuilder builder(call);
      SmallVector<Value> operands{function.getArgument(directoryIndex)};
      operands.append(call.getArgOperands().begin(),
                      call.getArgOperands().end());
      SmallVector<Attribute> args{builder.getIndexAttr(0)};
      if (ArrayAttr oldArgs = call.getArgsAttr()) {
        for (Attribute argument : oldArgs) {
          if (auto index = dyn_cast<IntegerAttr>(argument))
            args.push_back(builder.getIndexAttr(index.getInt() + 1));
          else
            args.push_back(argument);
        }
      } else {
        for (unsigned i = 1; i < operands.size(); ++i)
          args.push_back(builder.getIndexAttr(i));
      }
      auto replacement = CallOpaqueOp::create(
          builder, call.getLoc(), call.getResultTypes(), call.getCallee(),
          operands, builder.getArrayAttr(args), call.getTemplateArgsAttr());
      call.replaceAllUsesWith(replacement.getResults());
      call.erase();
    }
  }

  SmallVector<func::CallOp> callSites;
  module.walk([&](func::CallOp call) {
    if (auto callee = lookupCallee(call.getCallee());
        callee && needsDirectory.contains(callee.getOperation()))
      callSites.push_back(call);
  });
  for (func::CallOp call : callSites) {
    OpBuilder builder(call);
    SmallVector<Value> operands(call.getOperands());
    func::FuncOp caller = call->getParentOfType<func::FuncOp>();
    if (caller && needsDirectory.contains(caller.getOperation())) {
      operands.push_back(caller.getArguments().back());
    } else {
      operands.push_back(CallOpaqueOp::create(builder, call.getLoc(),
                                              TypeRange{directoryType},
                                              "std::string_view", ValueRange{})
                             .getResult(0));
    }
    auto replacement =
        func::CallOp::create(builder, call.getLoc(), call.getCallee(),
                             call.getResultTypes(), operands);
    replacement->setAttrs(call->getAttrs());
    call.replaceAllUsesWith(replacement.getResults());
    call.erase();
  }

  SmallVector<CallOpaqueOp> opaqueCallSites;
  module.walk([&](CallOpaqueOp call) {
    if (auto callee = lookupCallee(call.getCallee());
        callee && needsDirectory.contains(callee.getOperation()))
      opaqueCallSites.push_back(call);
  });
  for (CallOpaqueOp call : opaqueCallSites) {
    OpBuilder builder(call);
    SmallVector<Value> operands(call.getArgOperands());
    func::FuncOp caller = call->getParentOfType<func::FuncOp>();
    if (caller && needsDirectory.contains(caller.getOperation())) {
      operands.push_back(caller.getArguments().back());
    } else {
      operands.push_back(CallOpaqueOp::create(builder, call.getLoc(),
                                              TypeRange{directoryType},
                                              "std::string_view", ValueRange{})
                             .getResult(0));
    }
    SmallVector<Attribute> args;
    if (ArrayAttr oldArgs = call.getArgsAttr())
      args.append(oldArgs.begin(), oldArgs.end());
    else
      for (unsigned i = 0; i + 1 < operands.size(); ++i)
        args.push_back(builder.getIndexAttr(i));
    args.push_back(builder.getIndexAttr(operands.size() - 1));
    auto replacement = CallOpaqueOp::create(
        builder, call.getLoc(), call.getResultTypes(), call.getCallee(),
        operands, builder.getArrayAttr(args), call.getTemplateArgsAttr());
    call.replaceAllUsesWith(replacement.getResults());
    call.erase();
  }
  return success();
}

void addTupleAlias(OpBuilder& builder, Location loc, StringRef name,
                   ArrayRef<Type> fields) {
  emitVerbatim(builder, loc,
               "using " + name.str() + " = " + tupleTypeName(fields) + ";");
}

void addKeyPairClass(OpBuilder& builder, Location loc, Type storageType) {
  auto keyPair = ClassOp::create(builder, loc, "KeyPair");
  keyPair.getBody().emplaceBlock();
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(&keyPair.getBlock());
  FieldOp::create(builder, loc, "storage", storageType, Attribute{});
  FieldOp::create(builder, loc, "secret_key",
                  OpaqueType::get(builder.getContext(), "SecretKey"),
                  Attribute{});
  FieldOp::create(builder, loc, "public_key",
                  OpaqueType::get(builder.getContext(), "PublicKey"),
                  Attribute{});
}

void addPublicDeclarations(OpBuilder& builder, Location loc,
                           Type decryptResultType) {
  auto* ctx = builder.getContext();
  Type contextRef = OpaqueType::get(ctx, "Context&");
  Type setupResult = OpaqueType::get(ctx, "std::shared_ptr<Context>");
  Type keyPair = OpaqueType::get(ctx, "KeyPair");
  Type secretKey = OpaqueType::get(ctx, "SecretKey");
  Type publicKey = OpaqueType::get(ctx, "PublicKey");
  Type prepared = OpaqueType::get(ctx, "PreparedInputs");
  Type encryptedInputs = OpaqueType::get(ctx, "EncryptedInputs");
  Type encryptedOutputs = OpaqueType::get(ctx, "EncryptedOutputs");

  createEmitCFunction(builder, loc, "Setup", {}, {setupResult}, true);
  createEmitCFunction(builder, loc, "KeyGen",
                      {OpaqueType::get(ctx, "const std::shared_ptr<Context>&")},
                      {keyPair}, true);
  createEmitCFunction(
      builder, loc, "Preprocess",
      {contextRef, publicKey, OpaqueType::get(ctx, "std::string_view")},
      {prepared}, true);

  createEmitCFunction(
      builder, loc, "Encrypt",
      {contextRef, secretKey, OpaqueType::get(ctx, "CleartextInputs&")},
      {encryptedInputs}, true);
  createEmitCFunction(
      builder, loc, "Evaluate",
      {contextRef, publicKey, OpaqueType::get(ctx, "const PreparedInputs&"),
       OpaqueType::get(ctx, "const EncryptedInputs&")},
      {encryptedOutputs}, true);
  createEmitCFunction(
      builder, loc, "Decrypt",
      {contextRef, secretKey, OpaqueType::get(ctx, "const EncryptedOutputs&")},
      {decryptResultType}, true);
}

LogicalResult addSetupDefinition(OpBuilder& builder, Location loc,
                                 EntryFunctions& functions) {
  OpBuilder::InsertionGuard guard(builder);
  auto* ctx = builder.getContext();
  auto function = createEmitCFunction(
      builder, loc, "Setup", {},
      {OpaqueType::get(ctx, "std::shared_ptr<Context>")}, false);
  builder.setInsertionPointToStart(&function.getBody().front());
  Value context = createLocal(builder, loc, "std::shared_ptr<Context>");
  callInternal(builder, loc, functions.setup, context);
  Value result = moveValue(builder, loc, context, "std::shared_ptr<Context>");
  ReturnOp::create(builder, loc, result);
  return success();
}

LogicalResult addKeygenDefinition(OpBuilder& builder, Location loc,
                                  EntryFunctions& functions,
                                  Type keyStorageType) {
  OpBuilder::InsertionGuard guard(builder);
  auto* ctx = builder.getContext();
  auto function = createEmitCFunction(
      builder, loc, "KeyGen",
      {OpaqueType::get(ctx, "const std::shared_ptr<Context>&")},
      {OpaqueType::get(ctx, "KeyPair")}, false);
  builder.setInsertionPointToStart(&function.getBody().front());
  Value keyPair = createLocal(builder, loc, "KeyPair");
  Value storage = MemberOp::create(
      builder, loc, LValueType::get(keyStorageType), "storage", keyPair);
  callInternal(builder, loc, functions.keygen,
               ValueRange{function.getArgument(0), storage});
  Type uiPointer =
      PointerType::get(OpaqueType::get(ctx, "UserInterface<word>"));
  Value ui = CallOpaqueOp::create(builder, loc, TypeRange{uiPointer},
                                  "heir::getPointer", storage)
                 .getResult(0);
  for (StringRef field : {"secret_key", "public_key"}) {
    Type aliasType =
        OpaqueType::get(ctx, field == "secret_key" ? "SecretKey" : "PublicKey");
    Value member = MemberOp::create(builder, loc, LValueType::get(aliasType),
                                    field, keyPair);
    Value cast =
        CallOpaqueOp::create(builder, loc, TypeRange{aliasType},
                             "static_cast<" + cppTypeName(aliasType) + ">", ui)
            .getResult(0);
    emitc::AssignOp::create(builder, loc, member, cast);
  }
  ReturnOp::create(builder, loc, moveValue(builder, loc, keyPair, "KeyPair"));
  return success();
}

LogicalResult addPreprocessDefinition(OpBuilder& builder, Location loc,
                                      EntryFunctions& functions,
                                      ArrayRef<Type> preparedFields,
                                      bool takesResourceDirectory) {
  OpBuilder::InsertionGuard guard(builder);
  auto* ctx = builder.getContext();
  SmallVector<Type> inputs{OpaqueType::get(ctx, "Context&"),
                           OpaqueType::get(ctx, "PublicKey"),
                           OpaqueType::get(ctx, "std::string_view")};
  auto function =
      createEmitCFunction(builder, loc, "Preprocess", inputs,
                          {OpaqueType::get(ctx, "PreparedInputs")}, false);
  builder.setInsertionPointToStart(&function.getBody().front());
  SupportValues support = buildSupportValues(
      builder, loc, function.getArgument(0), function.getArgument(1));
  Value prepared = createLocal(builder, loc, "PreparedInputs");
  if (!functions.preprocess) {
    ReturnOp::create(builder, loc,
                     moveValue(builder, loc, prepared, "PreparedInputs"));
    return success();
  }
  SmallVector<Value> arguments;
  unsigned destination = 0;
  for (auto [index, type] :
       llvm::enumerate(functions.preprocess.getArgumentTypes())) {
    if (functions.preprocess.getArgAttr(index, "bufferize.result")) {
      arguments.push_back(getTupleElement(builder, loc, prepared, destination,
                                          preparedFields[destination]));
      ++destination;
      continue;
    }
    if (Value value = getSupportValue(type, support)) {
      arguments.push_back(value);
      continue;
    }
    if (takesResourceDirectory &&
        index + 1 == functions.preprocess.getNumArguments()) {
      arguments.push_back(function.getArgument(2));
      continue;
    }
    return functions.preprocess.emitOpError(
        "entry interface cannot source a preprocessing data argument");
  }
  callInternal(builder, loc, functions.preprocess, arguments);
  ReturnOp::create(builder, loc,
                   moveValue(builder, loc, prepared, "PreparedInputs"));
  return success();
}

LogicalResult addEncryptDefinition(OpBuilder& builder, Location loc,
                                   EntryFunctions& functions,
                                   ArrayRef<Type> logicalInputs,
                                   ArrayRef<Type> encryptedFields) {
  OpBuilder::InsertionGuard guard(builder);
  auto* ctx = builder.getContext();
  SmallVector<Type> inputs{OpaqueType::get(ctx, "Context&"),
                           OpaqueType::get(ctx, "SecretKey"),
                           OpaqueType::get(ctx, "CleartextInputs&")};
  auto function =
      createEmitCFunction(builder, loc, "Encrypt", inputs,
                          {OpaqueType::get(ctx, "EncryptedInputs")}, false);
  builder.setInsertionPointToStart(&function.getBody().front());
  SupportValues support = buildSupportValues(
      builder, loc, function.getArgument(0), function.getArgument(1));
  Value encrypted = createLocal(builder, loc, "EncryptedInputs");
  for (unsigned input = 0; input < logicalInputs.size(); ++input) {
    Type cleartextType = OpaqueType::get(ctx, "Input" + std::to_string(input));
    Value cleartext = getTupleElement(builder, loc, function.getArgument(2),
                                      input, cleartextType);
    Value field =
        getTupleElement(builder, loc, encrypted, input, encryptedFields[input]);
    func::FuncOp helper = findIndexedHelper(functions.inputHelpers, input);
    if (!helper) {
      if (isa<PointerType>(encryptedFields[input]))
        return functions.contract.emitOpError()
               << "entry input " << input
               << " requires an encryption or packing helper";
      Value copied =
          CallOpaqueOp::create(
              builder, loc, TypeRange{encryptedFields[input]},
              "static_cast<" + cppTypeName(encryptedFields[input]) + ">",
              cleartext)
              .getResult(0);
      emitc::AssignOp::create(builder, loc, field, copied);
      continue;
    }

    SmallVector<Value> arguments;
    unsigned dataArguments = 0;
    unsigned destinations = 0;
    for (auto [argumentIndex, type] :
         llvm::enumerate(helper.getArgumentTypes())) {
      if (helper.getArgAttr(argumentIndex, "bufferize.result")) {
        arguments.push_back(field);
        ++destinations;
        continue;
      }
      if (Value value = getSupportValue(type, support)) {
        arguments.push_back(value);
        continue;
      }
      arguments.push_back(getInputData(builder, loc, cleartext, type));
      ++dataArguments;
    }
    if (dataArguments != 1 || destinations > 1)
      return helper.emitOpError(
          "entry input helper must have one data argument and at most one "
          "destination");
    CallOpaqueOp call = callInternal(builder, loc, helper, arguments);
    if (destinations == 1) {
      if (!call.getResults().empty())
        return helper.emitOpError(
            "entry input helper cannot both return and store its result");
      continue;
    }
    if (call.getNumResults() != 1)
      return helper.emitOpError("entry input helper must produce one value");
    emitc::AssignOp::create(builder, loc, field, call.getResult(0));
  }
  ReturnOp::create(builder, loc,
                   moveValue(builder, loc, encrypted, "EncryptedInputs"));
  return success();
}

LogicalResult addEvaluateDefinition(OpBuilder& builder, Location loc,
                                    EntryFunctions& functions,
                                    ArrayRef<Type> preparedFields,
                                    ArrayRef<Type> encryptedInputFields,
                                    ArrayRef<Type> encryptedOutputFields) {
  OpBuilder::InsertionGuard guard(builder);
  auto* ctx = builder.getContext();
  auto function = createEmitCFunction(
      builder, loc, "Evaluate",
      {OpaqueType::get(ctx, "Context&"), OpaqueType::get(ctx, "PublicKey"),
       OpaqueType::get(ctx, "const PreparedInputs&"),
       OpaqueType::get(ctx, "const EncryptedInputs&")},
      {OpaqueType::get(ctx, "EncryptedOutputs")}, false);
  builder.setInsertionPointToStart(&function.getBody().front());
  SupportValues support = buildSupportValues(
      builder, loc, function.getArgument(0), function.getArgument(1));
  Value outputs = createLocal(builder, loc, "EncryptedOutputs");
  SmallVector<Value> arguments;
  unsigned encryptedInput = 0;
  unsigned preparedInput = 0;
  unsigned destination = 0;
  for (auto [index, type] :
       llvm::enumerate(functions.evaluate.getArgumentTypes())) {
    if (functions.evaluate.getArgAttr(index, "bufferize.result")) {
      arguments.push_back(getTupleElement(builder, loc, outputs, destination,
                                          encryptedOutputFields[destination]));
      ++destination;
      continue;
    }
    if (Value value = getSupportValue(type, support)) {
      arguments.push_back(value);
      continue;
    }
    if (encryptedInput < encryptedInputFields.size()) {
      arguments.push_back(
          getTupleElement(builder, loc, function.getArgument(3), encryptedInput,
                          encryptedInputFields[encryptedInput], true));
      ++encryptedInput;
      continue;
    }
    if (preparedInput < preparedFields.size()) {
      arguments.push_back(getTupleElement(builder, loc, function.getArgument(2),
                                          preparedInput,
                                          preparedFields[preparedInput], true));
      ++preparedInput;
      continue;
    }
    return functions.evaluate.emitOpError(
        "entry interface found an unmapped evaluation argument");
  }
  callInternal(builder, loc, functions.evaluate, arguments);
  ReturnOp::create(builder, loc,
                   moveValue(builder, loc, outputs, "EncryptedOutputs"));
  return success();
}

LogicalResult addDecryptDefinition(OpBuilder& builder, Location loc,
                                   EntryFunctions& functions,
                                   ArrayRef<Type> encryptedOutputFields,
                                   ArrayRef<std::string> logicalOutputNames,
                                   Type publicResultType) {
  OpBuilder::InsertionGuard guard(builder);
  auto* ctx = builder.getContext();
  auto function = createEmitCFunction(
      builder, loc, "Decrypt",
      {OpaqueType::get(ctx, "Context&"), OpaqueType::get(ctx, "SecretKey"),
       OpaqueType::get(ctx, "const EncryptedOutputs&")},
      {publicResultType}, false);
  builder.setInsertionPointToStart(&function.getBody().front());
  SupportValues support = buildSupportValues(
      builder, loc, function.getArgument(0), function.getArgument(1));
  SmallVector<Value> clearOutputs;
  for (unsigned index = 0; index < logicalOutputNames.size(); ++index)
    clearOutputs.push_back(
        createLocal(builder, loc, "Output" + std::to_string(index)));

  Value returnedOutputs;
  if (logicalOutputNames.size() > 1)
    returnedOutputs = createLocal(builder, loc, "Outputs");

  for (unsigned output = 0; output < logicalOutputNames.size(); ++output) {
    func::FuncOp helper = findIndexedHelper(functions.outputHelpers, output);
    if (!helper) {
      Value encryptedValue =
          getTupleElement(builder, loc, function.getArgument(2), output,
                          encryptedOutputFields[output], true);
      Value copied = CallOpaqueOp::create(
                         builder, loc,
                         TypeRange{OpaqueType::get(
                             ctx, "Output" + std::to_string(output))},
                         "static_cast<Output" + std::to_string(output) + ">",
                         encryptedValue)
                         .getResult(0);
      emitc::AssignOp::create(builder, loc, clearOutputs[output], copied);
      continue;
    }

    SmallVector<Value> arguments;
    unsigned encryptedArguments = 0;
    unsigned destinations = 0;
    for (auto [argumentIndex, type] :
         llvm::enumerate(helper.getArgumentTypes())) {
      if (helper.getArgAttr(argumentIndex, "bufferize.result")) {
        arguments.push_back(
            getInputData(builder, loc, clearOutputs[output], type));
        ++destinations;
        continue;
      }
      if (Value value = getSupportValue(type, support)) {
        arguments.push_back(value);
        continue;
      }
      arguments.push_back(getTupleElement(builder, loc, function.getArgument(2),
                                          output, encryptedOutputFields[output],
                                          true));
      ++encryptedArguments;
    }
    if (encryptedArguments != 1 || destinations > 1)
      return helper.emitOpError(
          "entry result helper must have one encrypted argument and at most "
          "one destination");
    CallOpaqueOp call = callInternal(builder, loc, helper, arguments);
    if (destinations == 1) {
      if (!call.getResults().empty())
        return helper.emitOpError(
            "entry result helper cannot both return and store its result");
      continue;
    }
    if (call.getNumResults() != 1)
      return helper.emitOpError("entry result helper must produce one value");
    emitc::AssignOp::create(builder, loc, clearOutputs[output],
                            call.getResult(0));
  }

  if (clearOutputs.size() == 1) {
    ReturnOp::create(builder, loc,
                     moveValue(builder, loc, clearOutputs[0], "Output0"));
    return success();
  }
  for (unsigned i = 0; i < clearOutputs.size(); ++i) {
    Value element =
        getTupleElement(builder, loc, returnedOutputs, i,
                        OpaqueType::get(ctx, logicalOutputNames[i]));
    emitc::AssignOp::create(
        builder, loc, element,
        moveValue(builder, loc, clearOutputs[i], "Output" + std::to_string(i)));
  }
  ReturnOp::create(builder, loc,
                   moveValue(builder, loc, returnedOutputs, "Outputs"));
  return success();
}

LogicalResult buildInterface(ModuleOp module, EntryFunctions& functions,
                             StringRef runtimeNamespace,
                             ArrayRef<StringRef> extensionIncludes) {
  Location loc = functions.setup.getLoc();
  MLIRContext* ctx = module.getContext();
  std::string runtimeNamespaceName = runtimeNamespace.str();

  ArrayAttr inputTypeAttrs =
      getLogicalTypes(functions.contract, kEntryInputTypesAttrName);
  ArrayAttr resultTypeAttrs =
      getLogicalTypes(functions.contract, kEntryResultTypesAttrName);
  if (!inputTypeAttrs || !resultTypeAttrs)
    return module.emitError("entry interface is missing logical type metadata");

  SmallVector<Type> logicalInputs;
  SmallVector<std::string> inputNames;
  for (Attribute attr : inputTypeAttrs) {
    Type type = cast<TypeAttr>(attr).getValue();
    FailureOr<std::string> name = logicalCppType(type, functions.contract);
    if (failed(name)) return failure();
    logicalInputs.push_back(type);
    inputNames.push_back(*name);
  }
  SmallVector<Type> logicalOutputs;
  SmallVector<std::string> outputNames;
  for (Attribute attr : resultTypeAttrs) {
    Type type = cast<TypeAttr>(attr).getValue();
    FailureOr<std::string> name = logicalCppType(type, functions.contract);
    if (failed(name)) return failure();
    logicalOutputs.push_back(type);
    outputNames.push_back(*name);
  }
  if (outputNames.empty())
    return module.emitError("void entry results are not yet supported");

  std::string contextName = contextTypeName(functions);
  if (contextName.empty())
    return module.emitError("entry interface could not identify its context");

  SmallVector<Type> setupDestinations = getDestinationTypes(functions.setup);
  SmallVector<Type> keygenDestinations = getDestinationTypes(functions.keygen);
  if (setupDestinations.size() != 1 || keygenDestinations.size() != 1)
    return module.emitError(
        "setup and key generation must each have one destination");
  SmallVector<Type> preparedFields;
  if (functions.preprocess)
    preparedFields = getDestinationTypes(functions.preprocess);
  SmallVector<Type> evaluationDataFields =
      getDataArgumentTypes(functions.evaluate);
  if (evaluationDataFields.size() !=
      logicalInputs.size() + preparedFields.size())
    return functions.evaluate.emitOpError()
           << "expected " << logicalInputs.size() << " entry inputs and "
           << preparedFields.size() << " prepared inputs, but found "
           << evaluationDataFields.size() << " data arguments";
  SmallVector<Type> encryptedInputFields(
      evaluationDataFields.begin(),
      evaluationDataFields.begin() + logicalInputs.size());
  SmallVector<Type> encryptedOutputFields =
      getDestinationTypes(functions.evaluate);
  if (encryptedOutputFields.size() != logicalOutputs.size())
    return functions.evaluate.emitOpError()
           << "expected " << logicalOutputs.size()
           << " entry result destinations, but found "
           << encryptedOutputFields.size();
  if (failed(validateIndexedHelpers(functions.inputHelpers,
                                    logicalInputs.size(), "entry input",
                                    functions.contract)) ||
      failed(validateIndexedHelpers(functions.outputHelpers,
                                    logicalOutputs.size(), "entry result",
                                    functions.contract)))
    return failure();

  bool takesResourceDirectory = false;
  if (functions.preprocess) {
    unsigned oldPreprocessArguments = functions.preprocess.getNumArguments();
    if (failed(addResourceDirectoryArguments(functions.preprocess)))
      return failure();
    takesResourceDirectory =
        functions.preprocess.getNumArguments() != oldPreprocessArguments;
  }

  OpBuilder builder(ctx);
  builder.setInsertionPointToEnd(module.getBody());
  FileOp header = FileOp::create(builder, loc, "header");
  FileOp source = FileOp::create(builder, loc, "source");

  builder.setInsertionPointToEnd(&header.getBodyRegion().front());
  emitVerbatim(builder, loc, "#pragma once");
  for (StringRef include : {"array", "complex", "cstddef", "cstdint", "memory",
                            "string_view", "tuple", "utility", "vector"})
    emitInclude(builder, loc, include);
  for (StringRef include : {"UserInterface.h", "core/Context.h",
                            "core/Encode.h", "core/Parameter.h"})
    emitInclude(builder, loc, include, false);
  for (StringRef include : extensionIncludes)
    emitInclude(builder, loc, include, false);

  std::string namespaceName =
      "heir::generated::" + sanitizeIdentifier(functions.entryName);
  emitVerbatim(builder, loc, "namespace " + namespaceName + " {");
  emitVerbatim(builder, loc, "using word = std::uint64_t;");
  emitVerbatim(builder, loc, "using Complex = std::complex<double>;");
  emitVerbatim(builder, loc, "using namespace ::" + runtimeNamespaceName + ";");
  emitVerbatim(
      builder, loc,
      "using Context = ::" + runtimeNamespaceName + "::" + contextName + ";");
  emitVerbatim(builder, loc,
               "using SecretKey = ::" + runtimeNamespaceName +
                   "::UserInterface<word>*;");
  emitVerbatim(builder, loc,
               "using PublicKey = ::" + runtimeNamespaceName +
                   "::UserInterface<word>*;");
  for (auto [index, name] : llvm::enumerate(inputNames))
    emitVerbatim(builder, loc,
                 "using Input" + std::to_string(index) + " = " + name + ";");
  SmallVector<Type> cleartextInputTypes;
  for (unsigned i = 0; i < inputNames.size(); ++i)
    cleartextInputTypes.push_back(
        OpaqueType::get(ctx, "Input" + std::to_string(i)));
  addTupleAlias(builder, loc, "CleartextInputs", cleartextInputTypes);
  for (auto [index, name] : llvm::enumerate(outputNames))
    emitVerbatim(builder, loc,
                 "using Output" + std::to_string(index) + " = " + name + ";");
  Type publicResultType = OpaqueType::get(ctx, "Output0");
  if (outputNames.size() > 1) {
    SmallVector<Type> outputAliasTypes;
    for (unsigned i = 0; i < outputNames.size(); ++i)
      outputAliasTypes.push_back(
          OpaqueType::get(ctx, "Output" + std::to_string(i)));
    addTupleAlias(builder, loc, "Outputs", outputAliasTypes);
    publicResultType = OpaqueType::get(ctx, "Outputs");
  }
  addKeyPairClass(builder, loc, keygenDestinations.front());
  addTupleAlias(builder, loc, "PreparedInputs", preparedFields);
  addTupleAlias(builder, loc, "EncryptedInputs", encryptedInputFields);
  addTupleAlias(builder, loc, "EncryptedOutputs", encryptedOutputFields);
  addPublicDeclarations(builder, loc, publicResultType);
  emitVerbatim(builder, loc, "}  // namespace " + namespaceName);

  builder.setInsertionPointToEnd(&source.getBodyRegion().front());
  emitInclude(builder, loc, functions.entryName + ".h", false);
  emitInclude(builder, loc, "lib/Runtime/CheddarRuntime.h", false);
  emitVerbatim(builder, loc, "namespace heir::generated::detail {");
  emitVerbatim(builder, loc, "using namespace ::" + runtimeNamespaceName + ";");
  emitVerbatim(builder, loc, "using word = std::uint64_t;");
  emitVerbatim(builder, loc, "using Complex = std::complex<double>;");

  SmallVector<Operation*> originalOperations;
  for (Operation& operation : module.getBody()->getOperations())
    if (&operation != header.getOperation() &&
        &operation != source.getOperation())
      originalOperations.push_back(&operation);
  for (Operation* operation : originalOperations) {
    if (isa<IncludeOp>(operation)) continue;
    if (auto function = dyn_cast<func::FuncOp>(operation))
      function.setPrivate();
    operation->moveBefore(&source.getBodyRegion().front(),
                          source.getBodyRegion().front().end());
  }
  builder.setInsertionPointToEnd(&source.getBodyRegion().front());
  emitVerbatim(builder, loc, "}  // namespace heir::generated::detail");
  emitVerbatim(builder, loc, "namespace " + namespaceName + " {");

  if (failed(addSetupDefinition(builder, loc, functions)) ||
      failed(addKeygenDefinition(builder, loc, functions,
                                 keygenDestinations.front())) ||
      failed(addPreprocessDefinition(builder, loc, functions, preparedFields,
                                     takesResourceDirectory)) ||
      failed(addEncryptDefinition(builder, loc, functions, logicalInputs,
                                  encryptedInputFields)) ||
      failed(addEvaluateDefinition(builder, loc, functions, preparedFields,
                                   encryptedInputFields,
                                   encryptedOutputFields)) ||
      failed(addDecryptDefinition(builder, loc, functions,
                                  encryptedOutputFields, outputNames,
                                  publicResultType)))
    return failure();
  emitVerbatim(builder, loc, "}  // namespace " + namespaceName);

  builder.setInsertionPointToStart(&source.getBodyRegion().front());
  for (Operation* operation : originalOperations)
    if (isa<IncludeOp>(operation))
      operation->moveBefore(&source.getBodyRegion().front(),
                            source.getBodyRegion().front().begin());
  return success();
}

struct CheddarEmitCEntryInterfacePass
    : public impl::CheddarEmitCEntryInterfaceBase<
          CheddarEmitCEntryInterfacePass> {
  using CheddarEmitCEntryInterfaceBase::CheddarEmitCEntryInterfaceBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    FailureOr<EntryFunctions> functions =
        findEntryFunctions(module, entryFunction);
    // The C++ facade exposes Setup and KeyGen separately, so both are required.
    if (succeeded(functions) && (!functions->setup || !functions->keygen)) {
      module.emitError() << "entry @" << functions->entryName
                         << " is missing a setup or keygen function";
      return signalPassFailure();
    }
    SmallVector<StringRef> extensionIncludes = {"extension/BootContext.h",
                                                "extension/EvalPoly.h",
                                                "extension/LinearTransform.h"};
    if (failed(functions) ||
        failed(
            buildInterface(module, *functions, "cheddar", extensionIncludes)))
      signalPassFailure();
  }
};

}  // namespace
}  // namespace mlir::heir
