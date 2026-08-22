#include <cctype>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>

#include "lib/Conversions/CheddarToEmitC/CheddarToEmitC.h"
#include "lib/Dialect/Cheddar/IR/CheddarTypes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Utils/EntryInterfaceUtils.h"
#include "mlir/include/mlir/Dialect/EmitC/IR/EmitC.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h" // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"           // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"            // from @llvm-project
#include "mlir/include/mlir/IR/TypeRange.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"   // from @llvm-project
#include "llvm/include/llvm/ADT/DenseSet.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/STLFunctionalExtras.h" // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"         // from @llvm-project
#include "llvm/include/llvm/ADT/StringExtras.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"           // from @llvm-project

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

// The lowered functions live in this namespace of the source file.
constexpr StringLiteral kDetailNamespace = "heir::generated::detail";

std::string sanitizeIdentifier(StringRef value) {
  std::string result;
  result.reserve(value.size());
  for (char c : value)
    result.push_back(std::isalnum(static_cast<unsigned char>(c)) ? c : '_');
  if (result.empty() || std::isdigit(static_cast<unsigned char>(result[0])))
    result.insert(result.begin(), '_');
  return result;
}

std::string trimReference(StringRef name) {
  name = name.trim();
  if (name.consume_front("const "))
    name = name.trim();
  if (name.consume_back("&"))
    name = name.rtrim();
  return name.str();
}

std::string stdArrayName(ArrayRef<int64_t> shape, StringRef element);

std::string cppTypeName(Type type) {
  if (auto opaque = dyn_cast<OpaqueType>(type))
    return opaque.getValue().str();
  if (auto array = dyn_cast<emitc::ArrayType>(type))
    return stdArrayName(array.getShape(), cppTypeName(array.getElementType()));
  if (auto pointer = dyn_cast<PointerType>(type))
    return cppTypeName(pointer.getPointee()) + "*";
  if (type.isF16())
    return "_Float16";
  if (type.isF32())
    return "float";
  if (type.isF64())
    return "double";
  if (type.isIndex())
    return "std::size_t";
  if (auto integer = dyn_cast<IntegerType>(type)) {
    if (integer.getWidth() == 1)
      return "bool";
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

FailureOr<std::string> logicalCppType(Type type, Operation *diagnostic) {
  if (auto tensor = dyn_cast<RankedTensorType>(type)) {
    if (!tensor.hasStaticShape())
      return diagnostic->emitError(
          "entry interface requires statically shaped logical tensors");
    std::string element = cppTypeName(tensor.getElementType());
    if (element.empty())
      return diagnostic->emitError()
             << "unsupported logical tensor element type "
             << tensor.getElementType();
    if (tensor.getRank() == 0)
      return element;
    return stdArrayName(tensor.getShape(), element);
  }
  std::string result = cppTypeName(type);
  if (result.empty())
    return diagnostic->emitError() << "unsupported logical type " << type;
  return result;
}

bool isDestination(func::FuncOp function, unsigned index) {
  return static_cast<bool>(function.getArgAttr(index, "bufferize.result"));
}

// The C++ type a facade stores for a lowered argument: references and const
// are dropped (a payload buffer argument is a `const T[N]` / `T[N]` array).
Type storedType(Type argumentType) {
  if (auto opaque = dyn_cast<OpaqueType>(argumentType))
    return OpaqueType::get(argumentType.getContext(),
                           trimReference(opaque.getValue()));
  if (auto array = dyn_cast<emitc::ArrayType>(argumentType))
    if (auto element = dyn_cast<OpaqueType>(array.getElementType()))
      return emitc::ArrayType::get(
          array.getShape(), OpaqueType::get(argumentType.getContext(),
                                            trimReference(element.getValue())));
  return argumentType;
}

SmallVector<Type> getDestinationTypes(func::FuncOp function) {
  SmallVector<Type> result;
  for (unsigned i = 0; i < function.getNumArguments(); ++i)
    if (isDestination(function, i))
      result.push_back(storedType(function.getArgumentTypes()[i]));
  return result;
}

unsigned countDestinations(func::FuncOp function) {
  unsigned count = 0;
  for (unsigned i = 0; i < function.getNumArguments(); ++i)
    count += isDestination(function, i);
  return count;
}

std::string tupleTypeName(ArrayRef<Type> fields) {
  std::string result = "std::tuple<";
  for (auto [index, type] : llvm::enumerate(fields)) {
    if (index)
      result += ", ";
    result += cppTypeName(type);
  }
  return result + ">";
}

// A cleartext scalar that reaches the server as-is is stored, not encrypted.
bool isScalarField(Type type) { return type.isIntOrIndexOrFloat(); }

//===----------------------------------------------------------------------===//
// Facade metadata (see cheddar-build-entry-interface)
//===----------------------------------------------------------------------===//

StringRef supportKindOf(func::FuncOp function, unsigned index) {
  auto kind = function.getArgAttrOfType<StringAttr>(
      index, cheddar::kSupportArgAttrName);
  return kind ? kind.getValue() : StringRef();
}

bool isPreparedArgument(func::FuncOp function, unsigned index) {
  return static_cast<bool>(
      function.getArgAttr(index, cheddar::kPreparedArgAttrName));
}

std::optional<int64_t> entryInputOf(func::FuncOp function, unsigned index) {
  auto input = function.getArgAttrOfType<IntegerAttr>(
      index, cheddar::kEntryInputArgAttrName);
  if (!input)
    return std::nullopt;
  return input.getInt();
}

bool isDataArgument(func::FuncOp function, unsigned index) {
  return !isDestination(function, index) &&
         supportKindOf(function, index).empty() &&
         !isPreparedArgument(function, index);
}

// The C++ type of the context a facade's caller owns: its first argument.
FailureOr<std::string> owningContextName(func::FuncOp facade) {
  StringRef kind =
      facade.getNumArguments() ? supportKindOf(facade, 0) : StringRef();
  if (kind == cheddar::ContextType::getMnemonic())
    return std::string("Context<word>");
  if (kind == cheddar::BootContextType::getMnemonic())
    return std::string("BootContext<word>");
  return facade.emitOpError("does not take its caller's context first");
}

bool needsSecret(func::FuncOp facade) {
  if (!facade)
    return false;
  for (unsigned i = 0; i < facade.getNumArguments(); ++i)
    if (supportKindOf(facade, i) == cheddar::UserInterfaceType::getMnemonic())
      return true;
  return false;
}

// The tuple fields of the public aggregates, read off the evaluate facade:
// its data arguments are the encrypted inputs, its prepared arguments the
// preprocessed values, its destinations the encrypted outputs.
struct AggregateFields {
  SmallVector<Type> inputs;
  SmallVector<Type> prepared;
  SmallVector<Type> outputs;
};

AggregateFields aggregateFields(func::FuncOp evaluate) {
  AggregateFields fields;
  for (auto [index, type] : llvm::enumerate(evaluate.getArgumentTypes())) {
    if (isDestination(evaluate, index))
      fields.outputs.push_back(storedType(type));
    else if (isPreparedArgument(evaluate, index))
      fields.prepared.push_back(storedType(type));
    else if (supportKindOf(evaluate, index).empty())
      fields.inputs.push_back(storedType(type));
  }
  return fields;
}

//===----------------------------------------------------------------------===//
// EmitC building blocks
//===----------------------------------------------------------------------===//

void emitVerbatim(OpBuilder &builder, Location loc, StringRef text) {
  VerbatimOp::create(builder, loc, text, ValueRange{});
}

void emitInclude(OpBuilder &builder, Location loc, StringRef include,
                 bool standard = true) {
  IncludeOp::create(builder, loc, include, standard);
}

FuncOp createEmitCFunction(OpBuilder &builder, Location loc, StringRef name,
                           TypeRange inputs, TypeRange results,
                           bool declaration) {
  auto type = FunctionType::get(builder.getContext(), inputs, results);
  auto function = FuncOp::create(builder, loc, name, type);
  if (!declaration)
    function.addEntryBlock();
  return function;
}

Value createLocal(OpBuilder &builder, Location loc, StringRef typeName) {
  Type type = OpaqueType::get(builder.getContext(), typeName);
  return VariableOp::create(builder, loc, LValueType::get(type),
                            OpaqueAttr::get(builder.getContext(), ""));
}

Value callOpaque(OpBuilder &builder, Location loc, StringRef resultType,
                 StringRef callee, ValueRange arguments) {
  return CallOpaqueOp::create(
             builder, loc,
             TypeRange{OpaqueType::get(builder.getContext(), resultType)},
             callee, arguments)
      .getResult(0);
}

Value moveValue(OpBuilder &builder, Location loc, Value value,
                StringRef resultType) {
  return callOpaque(builder, loc, resultType, "std::move", value);
}

// `std::get<index>(tuple)` as a const reference to the field.
Value getTupleElement(OpBuilder &builder, Location loc, Value tuple,
                      unsigned index, Type fieldType) {
  return callOpaque(builder, loc, "const " + cppTypeName(fieldType) + "&",
                    "std::get<" + std::to_string(index) + ">", tuple);
}

// Hands `input` to a lowered argument of `expectedType`: a buffer argument
// takes the data pointer, anything else the value itself.
Value getInputData(OpBuilder &builder, Location loc, Value input,
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

// Moves the locals holding the fields into the aggregate `typeName`.
Value packAggregate(OpBuilder &builder, Location loc, StringRef typeName,
                    ArrayRef<Value> locals) {
  auto *ctx = builder.getContext();
  return CallOpaqueOp::create(
             builder, loc, TypeRange{OpaqueType::get(ctx, typeName)},
             "heir::pack", locals, /*args=*/ArrayAttr{},
             builder.getArrayAttr({OpaqueAttr::get(ctx, typeName)}))
      .getResult(0);
}

CallOpaqueOp callInternal(OpBuilder &builder, Location loc,
                          func::FuncOp function, ValueRange arguments) {
  return CallOpaqueOp::create(builder, loc, function.getResultTypes(),
                              "::" + kDetailNamespace.str() +
                                  "::" + function.getSymName().str(),
                              arguments);
}

// The values a public function owns and hands to a facade's support
// arguments. Absent ones are errors when a facade asks for them.
struct OwnedSupport {
  Value context;   // Context&
  Value secret;    // UserInterface<word>*
  Value keys;      // const EvkMap<word>*
  Value directory; // std::string_view
  Value contextPointer;
};

// The UserInterface a SecretKey / PublicKey alias stands for.
Value userInterface(OpBuilder &builder, Location loc, Value key) {
  auto *ctx = builder.getContext();
  return CallOpaqueOp::create(builder, loc,
                              TypeRange{PointerType::get(
                                  OpaqueType::get(ctx, "UserInterface<word>"))},
                              "static_cast<UserInterface<word>*>", key)
      .getResult(0);
}

FailureOr<Value> supportOperand(OpBuilder &builder, Location loc,
                                StringRef kind, OwnedSupport &owned,
                                Operation *diagnostic) {
  auto *ctx = builder.getContext();
  if (kind == cheddar::ContextType::getMnemonic() ||
      kind == cheddar::BootContextType::getMnemonic()) {
    // One pointer serves every context kind: C++ converts it to the base
    // the callee declares.
    if (!owned.contextPointer)
      owned.contextPointer =
          CallOpaqueOp::create(
              builder, loc,
              TypeRange{PointerType::get(OpaqueType::get(ctx, "Context"))},
              "std::addressof", owned.context)
              .getResult(0);
    return owned.contextPointer;
  }
  if (kind == cheddar::UserInterfaceType::getMnemonic()) {
    if (!owned.secret)
      return diagnostic->emitError(
          "facade needs the secret key, which this side does not hold");
    return userInterface(builder, loc, owned.secret);
  }
  if (kind == cheddar::EvkMapType::getMnemonic()) {
    if (owned.keys)
      return callOpaque(builder, loc, "const EvkMap<word>&", "heir::deref",
                        owned.keys);
    if (!owned.secret)
      return diagnostic->emitError("facade needs evaluation keys");
    Value ui = userInterface(builder, loc, owned.secret);
    return MemberCallOpaqueOp::create(
               builder, loc,
               TypeRange{OpaqueType::get(ctx, "const EvkMap<word>&")}, ui,
               "GetEvkMap", ArrayAttr{}, ArrayAttr{}, ValueRange{})
        .getResult(0);
  }
  if (kind == cheddar::kResourceDirSupportKind) {
    // A step the interface hands no directory reads resources relative to
    // the working directory.
    if (!owned.directory)
      owned.directory =
          callOpaque(builder, loc, "std::string_view", "std::string_view", {});
    return owned.directory;
  }
  return diagnostic->emitError()
         << "facade needs an unknown support value " << kind;
}

// Calls `facade`: support arguments from `owned`, data arguments from
// `data(argumentIndex)`, destinations from `destination(k)` for the k-th one.
LogicalResult callFacade(OpBuilder &builder, Location loc, func::FuncOp facade,
                         OwnedSupport &owned,
                         llvm::function_ref<Value(unsigned)> data,
                         llvm::function_ref<Value(unsigned)> destination) {
  SmallVector<Value> operands;
  unsigned destinations = 0;
  for (auto [index, type] : llvm::enumerate(facade.getArgumentTypes())) {
    if (isDestination(facade, index)) {
      operands.push_back(
          getInputData(builder, loc, destination(destinations++), type));
      continue;
    }
    StringRef kind = supportKindOf(facade, index);
    if (!kind.empty()) {
      FailureOr<Value> value =
          supportOperand(builder, loc, kind, owned, facade);
      if (failed(value))
        return failure();
      operands.push_back(*value);
      continue;
    }
    operands.push_back(getInputData(builder, loc, data(index), type));
  }
  callInternal(builder, loc, facade, operands);
  return success();
}

//===----------------------------------------------------------------------===//
// Public types and functions
//===----------------------------------------------------------------------===//

void addTupleAlias(OpBuilder &builder, Location loc, StringRef name,
                   ArrayRef<Type> fields) {
  emitVerbatim(builder, loc,
               "using " + name.str() + " = " + tupleTypeName(fields) + ";");
}

void addKeyPairClass(OpBuilder &builder, Location loc, Type storageType) {
  auto keyPair =
      ClassOp::create(builder, loc, "KeyPair", /*sym_visibility=*/StringAttr{});
  keyPair.getBody().emplaceBlock();
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(&keyPair.getBlock());
  FieldOp::create(builder, loc, "storage", /*sym_visibility=*/StringAttr{},
                  storageType, Attribute{});
  FieldOp::create(builder, loc, "secret_key", /*sym_visibility=*/StringAttr{},
                  OpaqueType::get(builder.getContext(), "SecretKey"),
                  Attribute{});
  FieldOp::create(builder, loc, "public_key", /*sym_visibility=*/StringAttr{},
                  OpaqueType::get(builder.getContext(), "PublicKey"),
                  Attribute{});
}

LogicalResult addSetupDefinition(OpBuilder &builder, Location loc,
                                 EntryFunctions &functions) {
  OpBuilder::InsertionGuard guard(builder);
  auto *ctx = builder.getContext();
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

LogicalResult addKeygenDefinition(OpBuilder &builder, Location loc,
                                  EntryFunctions &functions,
                                  Type keyStorageType,
                                  bool evaluationNeedsSecret) {
  OpBuilder::InsertionGuard guard(builder);
  auto *ctx = builder.getContext();
  SmallVector<Type> inputs{
      OpaqueType::get(ctx, "const std::shared_ptr<Context>&")};
  auto function = createEmitCFunction(builder, loc, "KeyGen", inputs,
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
    Value value;
    if (field == "public_key" && !evaluationNeedsSecret) {
      // The map lives inside the UserInterface `storage` owns, so the pointer
      // is good for as long as the KeyPair is.
      Value map = MemberCallOpaqueOp::create(
                      builder, loc,
                      TypeRange{OpaqueType::get(ctx, "const EvkMap<word>&")},
                      ui, "GetEvkMap", ArrayAttr{}, ArrayAttr{}, ValueRange{})
                      .getResult(0);
      value = CallOpaqueOp::create(builder, loc, TypeRange{aliasType},
                                   "std::addressof", map)
                  .getResult(0);
    } else {
      value = CallOpaqueOp::create(
                  builder, loc, TypeRange{aliasType},
                  "static_cast<" + cppTypeName(aliasType) + ">", ui)
                  .getResult(0);
    }
    emitc::AssignOp::create(builder, loc, member, value);
  }
  ReturnOp::create(builder, loc, moveValue(builder, loc, keyPair, "KeyPair"));
  return success();
}

// Everything a public function definition needs about its side.
struct WrapperContext {
  EntryFunctions &functions;
  AggregateFields fields;
  // The public parameter carrying the server's key material is a
  // UserInterface (needsSecret) or an evaluation-key map.
  bool needsSecret;
};

// Preprocess(Context&, PublicKey, std::string_view) -> PreparedInputs
LogicalResult addPreprocessDefinition(OpBuilder &builder, Location loc,
                                      WrapperContext &wrapper) {
  OpBuilder::InsertionGuard guard(builder);
  auto *ctx = builder.getContext();
  SmallVector<Type> inputs{OpaqueType::get(ctx, "Context&"),
                           OpaqueType::get(ctx, "PublicKey"),
                           OpaqueType::get(ctx, "std::string_view")};
  auto function =
      createEmitCFunction(builder, loc, "Preprocess", inputs,
                          {OpaqueType::get(ctx, "PreparedInputs")}, false);
  builder.setInsertionPointToStart(&function.getBody().front());
  OwnedSupport owned;
  owned.context = function.getArgument(0);
  (wrapper.needsSecret ? owned.secret : owned.keys) = function.getArgument(1);
  owned.directory = function.getArgument(2);

  func::FuncOp facade = wrapper.functions.facadePreprocess;
  SmallVector<Value> locals;
  for (Type field : wrapper.fields.prepared)
    locals.push_back(createLocal(builder, loc, cppTypeName(field)));
  if (facade) {
    if (countDestinations(facade) != locals.size())
      return facade.emitOpError() << "produces " << countDestinations(facade)
                                  << " values, but the evaluate facade takes "
                                  << locals.size() << " preprocessed inputs";
    for (unsigned i = 0; i < facade.getNumArguments(); ++i)
      if (isDataArgument(facade, i))
        return facade.emitOpError(
            "takes a data argument the interface cannot supply");
    if (failed(callFacade(
            builder, loc, facade, owned, [](unsigned) { return Value(); },
            [&](unsigned k) { return locals[k]; })))
      return failure();
  } else if (!locals.empty()) {
    return wrapper.functions.facadeEvaluate.emitOpError(
        "takes preprocessed inputs, but there is no preprocess facade");
  }
  ReturnOp::create(builder, loc,
                   packAggregate(builder, loc, "PreparedInputs", locals));
  return success();
}

// Encrypt(Context&, SecretKey, CleartextInputs&) -> EncryptedInputs
LogicalResult addEncryptDefinition(OpBuilder &builder, Location loc,
                                   WrapperContext &wrapper) {
  OpBuilder::InsertionGuard guard(builder);
  auto *ctx = builder.getContext();
  auto function = createEmitCFunction(
      builder, loc, "Encrypt",
      {OpaqueType::get(ctx, "Context&"), OpaqueType::get(ctx, "SecretKey"),
       OpaqueType::get(ctx, "CleartextInputs&")},
      {OpaqueType::get(ctx, "EncryptedInputs")}, false);
  builder.setInsertionPointToStart(&function.getBody().front());
  OwnedSupport owned;
  owned.context = function.getArgument(0);
  owned.secret = function.getArgument(1);
  Value cleartext = function.getArgument(2);
  func::FuncOp evaluate = wrapper.functions.facadeEvaluate;
  func::FuncOp facade = wrapper.functions.facadeEncrypt;

  auto cleartextInput = [&](int64_t input) {
    std::string alias = "Input" + std::to_string(input);
    return callOpaque(builder, loc, alias + "&",
                      "std::get<" + std::to_string(input) + ">", cleartext);
  };
  for (unsigned i = 0; i < facade.getNumArguments(); ++i)
    if (isDataArgument(facade, i) && !entryInputOf(facade, i))
      return facade.emitOpError()
             << "argument " << i << " has no entry input to encrypt";

  // One local per encrypted-input field. A scalar field copies its cleartext
  // input; the others are the encrypt facade's destinations, in order.
  SmallVector<Value> locals;
  SmallVector<Value> encrypted;
  unsigned field = 0;
  for (auto [index, type] : llvm::enumerate(evaluate.getArgumentTypes())) {
    if (!isDataArgument(evaluate, index))
      continue;
    Type stored = wrapper.fields.inputs[field++];
    Value local = createLocal(builder, loc, cppTypeName(stored));
    locals.push_back(local);
    if (!isScalarField(stored)) {
      encrypted.push_back(local);
      continue;
    }
    std::optional<int64_t> input = entryInputOf(evaluate, index);
    if (!input)
      return evaluate.emitOpError()
             << "argument " << index
             << " is a cleartext scalar with no entry input";
    Value copied = callOpaque(builder, loc, cppTypeName(stored),
                              "static_cast<" + cppTypeName(stored) + ">",
                              cleartextInput(*input));
    emitc::AssignOp::create(builder, loc, local, copied);
  }
  if (countDestinations(facade) != encrypted.size())
    return facade.emitOpError() << "encrypts " << countDestinations(facade)
                                << " values, but the evaluate facade takes "
                                << encrypted.size() << " encrypted inputs";
  if (failed(callFacade(
          builder, loc, facade, owned,
          [&](unsigned index) {
            return cleartextInput(*entryInputOf(facade, index));
          },
          [&](unsigned k) { return encrypted[k]; })))
    return failure();
  ReturnOp::create(builder, loc,
                   packAggregate(builder, loc, "EncryptedInputs", locals));
  return success();
}

// Evaluate(Context&, PublicKey, const PreparedInputs&, const EncryptedInputs&)
//     -> EncryptedOutputs
LogicalResult addEvaluateDefinition(OpBuilder &builder, Location loc,
                                    WrapperContext &wrapper) {
  OpBuilder::InsertionGuard guard(builder);
  auto *ctx = builder.getContext();
  SmallVector<Type> inputs{OpaqueType::get(ctx, "Context&"),
                           OpaqueType::get(ctx, "PublicKey"),
                           OpaqueType::get(ctx, "const PreparedInputs&"),
                           OpaqueType::get(ctx, "const EncryptedInputs&")};
  auto function =
      createEmitCFunction(builder, loc, "Evaluate", inputs,
                          {OpaqueType::get(ctx, "EncryptedOutputs")}, false);
  builder.setInsertionPointToStart(&function.getBody().front());
  OwnedSupport owned;
  owned.context = function.getArgument(0);
  (wrapper.needsSecret ? owned.secret : owned.keys) = function.getArgument(1);
  Value prepared = function.getArgument(2);
  Value encrypted = function.getArgument(3);
  func::FuncOp facade = wrapper.functions.facadeEvaluate;

  SmallVector<Value> locals;
  for (Type field : wrapper.fields.outputs)
    locals.push_back(createLocal(builder, loc, cppTypeName(field)));
  unsigned input = 0;
  unsigned preparedInput = 0;
  if (failed(callFacade(
          builder, loc, facade, owned,
          [&](unsigned index) {
            if (isPreparedArgument(facade, index)) {
              unsigned k = preparedInput++;
              return getTupleElement(builder, loc, prepared, k,
                                     wrapper.fields.prepared[k]);
            }
            unsigned k = input++;
            return getTupleElement(builder, loc, encrypted, k,
                                   wrapper.fields.inputs[k]);
          },
          [&](unsigned k) { return locals[k]; })))
    return failure();
  ReturnOp::create(builder, loc,
                   packAggregate(builder, loc, "EncryptedOutputs", locals));
  return success();
}

// Decrypt(Context&, SecretKey, const EncryptedOutputs&) -> Output0 | Outputs
LogicalResult addDecryptDefinition(OpBuilder &builder, Location loc,
                                   WrapperContext &wrapper,
                                   ArrayRef<std::string> logicalOutputNames,
                                   Type publicResultType) {
  OpBuilder::InsertionGuard guard(builder);
  auto *ctx = builder.getContext();
  auto function = createEmitCFunction(
      builder, loc, "Decrypt",
      {OpaqueType::get(ctx, "Context&"), OpaqueType::get(ctx, "SecretKey"),
       OpaqueType::get(ctx, "const EncryptedOutputs&")},
      {publicResultType}, false);
  builder.setInsertionPointToStart(&function.getBody().front());
  OwnedSupport owned;
  owned.context = function.getArgument(0);
  owned.secret = function.getArgument(1);
  Value encrypted = function.getArgument(2);
  func::FuncOp facade = wrapper.functions.facadeDecrypt;

  SmallVector<Value> locals;
  for (unsigned index = 0; index < logicalOutputNames.size(); ++index)
    locals.push_back(
        createLocal(builder, loc, "Output" + std::to_string(index)));
  if (countDestinations(facade) != locals.size())
    return facade.emitOpError()
           << "decrypts " << countDestinations(facade) << " values for "
           << locals.size() << " entry results";
  unsigned output = 0;
  if (failed(callFacade(
          builder, loc, facade, owned,
          [&](unsigned) {
            unsigned k = output++;
            return getTupleElement(builder, loc, encrypted, k,
                                   wrapper.fields.outputs[k]);
          },
          [&](unsigned k) { return locals[k]; })))
    return failure();
  if (locals.size() == 1) {
    ReturnOp::create(builder, loc,
                     moveValue(builder, loc, locals[0], "Output0"));
    return success();
  }
  ReturnOp::create(builder, loc,
                   packAggregate(builder, loc, "Outputs", locals));
  return success();
}

//===----------------------------------------------------------------------===//
// Files
//===----------------------------------------------------------------------===//

LogicalResult buildInterface(ModuleOp module, EntryFunctions functions,
                             ArrayRef<StringRef> extensionIncludes) {
  Location loc = functions.setup.getLoc();
  MLIRContext *ctx = module.getContext();
  const std::string runtimeNamespaceName = "cheddar";

  if (!functions.facadeEvaluate || !functions.facadeEncrypt ||
      !functions.facadeDecrypt)
    return module.emitError(
        "entry interface requires the encrypt, decrypt and evaluate facades; "
        "cheddar-build-entry-interface did not generate them (see its "
        "warnings)");
  ArrayAttr inputTypeAttrs =
      getLogicalTypes(functions.contract, kEntryInputTypes);
  ArrayAttr resultTypeAttrs =
      getLogicalTypes(functions.contract, kEntryResultTypes);
  if (!inputTypeAttrs || !resultTypeAttrs)
    return module.emitError("entry interface is missing logical type metadata");

  SmallVector<std::string> inputNames;
  for (Attribute attr : inputTypeAttrs) {
    FailureOr<std::string> name =
        logicalCppType(cast<TypeAttr>(attr).getValue(), functions.contract);
    if (failed(name))
      return failure();
    inputNames.push_back(*name);
  }
  SmallVector<std::string> outputNames;
  for (Attribute attr : resultTypeAttrs) {
    FailureOr<std::string> name =
        logicalCppType(cast<TypeAttr>(attr).getValue(), functions.contract);
    if (failed(name))
      return failure();
    outputNames.push_back(*name);
  }
  if (outputNames.empty())
    return module.emitError("void entry results are not yet supported");

  // The caller owns the context the facades were built for.
  FailureOr<std::string> contextName =
      owningContextName(functions.facadeEncrypt);
  if (failed(contextName))
    return failure();
  // The server holds evaluation keys only, unless evaluation must decrypt
  // (e.g. debug mode).
  bool serverNeedsSecret = needsSecret(functions.facadeEvaluate) ||
                           needsSecret(functions.facadePreprocess);

  SmallVector<Type> setupDestinations = getDestinationTypes(functions.setup);
  SmallVector<Type> keygenDestinations = getDestinationTypes(functions.keygen);
  if (setupDestinations.size() != 1 || keygenDestinations.size() != 1)
    return module.emitError(
        "setup and key generation must each have one destination");
  WrapperContext wrapper{functions, aggregateFields(functions.facadeEvaluate),
                         serverNeedsSecret};

  OpBuilder builder(ctx);
  builder.setInsertionPointToEnd(module.getBody());
  FileOp header = FileOp::create(builder, loc, "header");
  FileOp source = FileOp::create(builder, loc, "source");

  builder.setInsertionPointToEnd(&header.getBodyRegion().front());
  emitVerbatim(builder, loc, "#pragma once");
  for (StringRef include : {"array", "complex", "cstddef", "cstdint", "memory",
                            "string_view", "tuple", "utility", "vector"})
    emitInclude(builder, loc, include);
  emitInclude(builder, loc, "UserInterface.h", false);
  emitInclude(builder, loc, "core/Context.h", false);
  emitInclude(builder, loc, "core/Encode.h", false);
  emitInclude(builder, loc, "core/Parameter.h", false);
  for (StringRef include : extensionIncludes)
    emitInclude(builder, loc, include, false);

  std::string namespaceName =
      "heir::generated::" + sanitizeIdentifier(functions.entryName);
  std::string detailNamespace = kDetailNamespace.str();
  emitVerbatim(builder, loc, "namespace " + namespaceName + " {");
  emitVerbatim(builder, loc, "using word = std::uint64_t;");
  emitVerbatim(builder, loc, "using Complex = std::complex<double>;");
  emitVerbatim(builder, loc, "using namespace ::" + runtimeNamespaceName + ";");
  emitVerbatim(builder, loc,
               "using Context = ::" + runtimeNamespaceName +
                   "::" + *contextName + ";");
  emitVerbatim(builder, loc,
               "using SecretKey = ::" + runtimeNamespaceName +
                   "::UserInterface<word>*;");
  emitVerbatim(
      builder, loc,
      "using PublicKey = " +
          std::string(
              serverNeedsSecret
                  ? "::" + runtimeNamespaceName + "::UserInterface<word>*;"
                  : "const ::" + runtimeNamespaceName + "::EvkMap<word>*;"));
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
  addTupleAlias(builder, loc, "PreparedInputs", wrapper.fields.prepared);
  addTupleAlias(builder, loc, "EncryptedInputs", wrapper.fields.inputs);
  addTupleAlias(builder, loc, "EncryptedOutputs", wrapper.fields.outputs);
  auto headerEnd = VerbatimOp::create(
      builder, loc, "}  // namespace " + namespaceName, ValueRange{});

  builder.setInsertionPointToEnd(&source.getBodyRegion().front());
  emitInclude(builder, loc, functions.entryName + ".h", false);
  emitInclude(builder, loc, "heir/runtime/CheddarRuntime.h", false);
  emitVerbatim(builder, loc, "namespace " + detailNamespace + " {");
  emitVerbatim(builder, loc, "using namespace ::" + runtimeNamespaceName + ";");
  emitVerbatim(builder, loc, "using word = std::uint64_t;");
  emitVerbatim(builder, loc, "using Complex = std::complex<double>;");

  // This side's source holds what its public functions reach: the facades,
  // the helpers they call, and the globals those read.
  llvm::DenseSet<Operation *> reachable;
  std::function<void(func::FuncOp)> visit = [&](func::FuncOp function) {
    if (!function || !reachable.insert(function).second)
      return;
    function.walk([&](func::CallOp call) {
      visit(module.lookupSymbol<func::FuncOp>(call.getCallee()));
    });
    function.walk([&](CallOpaqueOp call) {
      visit(module.lookupSymbol<func::FuncOp>(call.getCallee()));
    });
    function.walk([&](emitc::GetGlobalOp access) {
      if (auto global = module.lookupSymbol<emitc::GlobalOp>(access.getName()))
        reachable.insert(global);
    });
  };
  visit(functions.setup);
  visit(functions.keygen);
  visit(functions.facadeEncrypt);
  visit(functions.facadeDecrypt);
  visit(functions.facadePreprocess);
  visit(functions.facadeEvaluate);
  // A shared helper is cloned into both files, without cloning two complete
  // intermediate modules.
  SmallVector<Operation *> originalOperations;
  for (Operation &operation : module.getBody()->getOperations())
    if (!isa<FileOp>(operation))
      originalOperations.push_back(&operation);
  for (Operation *operation : originalOperations) {
    if (isa<IncludeOp>(operation))
      continue;
    if (isa<func::FuncOp, emitc::GlobalOp>(operation) &&
        !reachable.contains(operation))
      continue;
    Operation *emitted = builder.clone(*operation);
    if (auto function = dyn_cast<func::FuncOp>(emitted))
      function.setPrivate();
    emitted->moveBefore(&source.getBodyRegion().front(),
                        source.getBodyRegion().front().end());
  }
  builder.setInsertionPointToEnd(&source.getBodyRegion().front());
  emitVerbatim(builder, loc, "}  // namespace " + detailNamespace);
  emitVerbatim(builder, loc, "namespace " + namespaceName + " {");

  if (failed(addSetupDefinition(builder, loc, functions)) ||
      failed(addKeygenDefinition(builder, loc, functions,
                                 keygenDestinations.front(),
                                 serverNeedsSecret)) ||
      failed(addEncryptDefinition(builder, loc, wrapper)) ||
      failed(addPreprocessDefinition(builder, loc, wrapper)) ||
      failed(addEvaluateDefinition(builder, loc, wrapper)) ||
      failed(addDecryptDefinition(builder, loc, wrapper, outputNames,
                                  publicResultType)))
    return failure();
  emitVerbatim(builder, loc, "}  // namespace " + namespaceName);

  // The header declares the public functions: the definitions without their
  // bodies, since a prototype cannot refer across `emitc.file`s.
  builder.setInsertionPoint(headerEnd);
  for (FuncOp wrapper : source.getBodyRegion().front().getOps<FuncOp>()) {
    auto declaration = cast<FuncOp>(builder.clone(*wrapper));
    declaration.getBody().dropAllReferences();
    declaration.getBody().getBlocks().clear();
  }

  builder.setInsertionPointToStart(&source.getBodyRegion().front());
  for (Operation *operation : originalOperations)
    if (isa<IncludeOp>(operation))
      builder.clone(*operation);
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
    if (failed(functions))
      return signalPassFailure();
    // The C++ facade exposes Setup and KeyGen separately, so both are required.
    if (!functions->setup || !functions->keygen) {
      module.emitError() << "entry @" << functions->entryName
                         << " is missing a setup or keygen function";
      return signalPassFailure();
    }
    if (!functions->contract) {
      module.emitError() << "entry @" << functions->entryName
                         << " has no cleartext contract to build an interface "
                            "for";
      return signalPassFailure();
    }
    if (failed(
            buildInterface(module, *functions,
                           {"extension/BootContext.h", "extension/EvalPoly.h",
                            "extension/LinearTransform.h"})))
      return signalPassFailure();
    // Only the files remain; whatever no public function reaches is dropped.
    for (Operation &op :
         llvm::make_early_inc_range(module.getBody()->getOperations()))
      if (!isa<FileOp>(op))
        op.erase();
  }
};

} // namespace
} // namespace mlir::heir
