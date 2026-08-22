#include "lib/Dialect/Cheddar/Transforms/BuildEntryInterface.h"

#include <string>
#include <utility>

#include "lib/Dialect/Cheddar/IR/CheddarOps.h"
#include "lib/Dialect/Cheddar/IR/CheddarTypes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/Preprocessing/IR/PreprocessingTypes.h"
#include "lib/Utils/EntryInterfaceUtils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/STLFunctionalExtras.h"  // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"          // from @llvm-project
#include "llvm/include/llvm/ADT/StringMap.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"            // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"              // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"              // from @llvm-project
#include "mlir/include/mlir/IR/TypeRange.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project

namespace mlir::heir::cheddar {

#define GEN_PASS_DEF_CHEDDARBUILDENTRYINTERFACE
#include "lib/Dialect/Cheddar/Transforms/BuildEntryInterface.h.inc"

namespace {

// The support kind an argument type provides, or empty for a data argument.
StringRef supportKind(Type type) {
  if (isa<preprocessing::ResourceDirType>(type)) return kResourceDirSupportKind;
  return getSupportKind(type);
}

bool isContextKind(StringRef kind) {
  return kind == ContextType::getMnemonic() ||
         kind == BootContextType::getMnemonic();
}

// A data argument of the facade.
struct DataArgument {
  Type type;
  SmallVector<NamedAttribute> attrs;
};

// Builds one facade. Its signature is the owned support values every called
// helper needs, then the data arguments; the body derives the remaining
// support values and calls the helpers.
class FacadeBuilder {
 public:
  FacadeBuilder(ModuleOp module, Location loc, StringRef name, StringRef role,
                StringRef entryName, Type primaryContext)
      : module(module),
        loc(loc),
        name(name),
        role(role),
        entryName(entryName),
        primaryContext(primaryContext),
        builder(module.getContext()) {}

  // Declares the support kinds the helper's signature needs.
  void requireSupportOf(func::FuncOp helper) {
    for (Type type : helper.getArgumentTypes()) {
      StringRef kind = supportKind(type);
      if (!kind.empty() && !llvm::is_contained(required, kind))
        required.push_back(kind);
    }
  }

  void addDataArgument(Type type, ArrayRef<NamedAttribute> attrs = {}) {
    dataArguments.push_back({type, SmallVector<NamedAttribute>(attrs)});
  }

  // Creates the facade with the arguments declared so far. `body` receives
  // the data arguments and returns the facade's results.
  FailureOr<func::FuncOp> build(
      llvm::function_ref<FailureOr<SmallVector<Value>>(ValueRange)> body) {
    MLIRContext* ctx = module.getContext();
    // Owned support values, in a fixed order: the primary context, further
    // context kinds, the key, the resource directory.
    SmallVector<Type> types;
    SmallVector<DictionaryAttr> attrs;
    auto support = [&](StringRef kind, Type type) {
      supportIndex[kind] = types.size();
      types.push_back(type);
      attrs.push_back(DictionaryAttr::get(
          ctx, {NamedAttribute(StringAttr::get(ctx, kSupportArgAttrName),
                               StringAttr::get(ctx, kind))}));
    };
    support(getSupportKind(primaryContext), primaryContext);
    for (Type type :
         {Type(ContextType::get(ctx)), Type(BootContextType::get(ctx))}) {
      StringRef kind = getSupportKind(type);
      if (type != primaryContext && llvm::is_contained(required, kind))
        support(kind, type);
    }
    bool needsSecret =
        llvm::is_contained(required, UserInterfaceType::getMnemonic());
    if (needsSecret) {
      support(UserInterfaceType::getMnemonic(), UserInterfaceType::get(ctx));
    } else if (llvm::is_contained(required, EvkMapType::getMnemonic()) ||
               llvm::is_contained(required, EvalKeyType::getMnemonic())) {
      support(EvkMapType::getMnemonic(), EvkMapType::get(ctx));
    }
    if (llvm::is_contained(required, kResourceDirSupportKind))
      support(kResourceDirSupportKind,
              preprocessing::ResourceDirType::get(ctx));
    unsigned dataStart = types.size();
    for (DataArgument& argument : dataArguments) {
      types.push_back(argument.type);
      attrs.push_back(DictionaryAttr::get(ctx, argument.attrs));
    }

    builder.setInsertionPointToEnd(module.getBody());
    function = func::FuncOp::create(builder, loc, name,
                                    FunctionType::get(ctx, types, TypeRange{}));
    function.setPublic();
    function.setAllArgAttrs(attrs);
    setInterfaceRole(
        function, role,
        builder.getDictionaryAttr({builder.getNamedAttr(
            kClientHelperFuncName, builder.getStringAttr(entryName))}));
    builder.setInsertionPointToStart(function.addEntryBlock());

    FailureOr<SmallVector<Value>> results =
        body(function.getArguments().drop_front(dataStart));
    if (failed(results)) {
      function.erase();
      return failure();
    }
    func::ReturnOp::create(builder, loc, *results);
    function.setType(
        FunctionType::get(ctx, types, TypeRange(ValueRange(*results))));
    return function;
  }

  // Calls `helper`, sourcing its support arguments from the facade and its
  // data arguments from `data`, in order.
  FailureOr<ValueRange> call(func::FuncOp helper, ValueRange data) {
    SmallVector<Value> operands;
    unsigned next = 0;
    for (Type type : helper.getArgumentTypes()) {
      StringRef kind = supportKind(type);
      if (kind.empty()) {
        if (next == data.size())
          return helper.emitOpError()
                 << "expects more data arguments than the facade supplies";
        operands.push_back(data[next++]);
        continue;
      }
      FailureOr<Value> value = supportValue(kind, helper);
      if (failed(value)) return failure();
      operands.push_back(*value);
    }
    if (next != data.size())
      return helper.emitOpError() << "takes " << next << " data arguments, but "
                                  << data.size() << " were supplied";
    return func::CallOp::create(builder, loc, helper, operands).getResults();
  }

 private:
  // The facade-level value for a support kind: an owned argument, or one
  // derived from the owned arguments.
  FailureOr<Value> supportValue(StringRef kind, func::FuncOp helper) {
    auto argument = supportIndex.find(kind);
    if (argument != supportIndex.end())
      return function.getArgument(argument->second);
    MLIRContext* ctx = module.getContext();
    if (kind == EncoderType::getMnemonic()) {
      if (!encoder)
        encoder = GetEncoderOp::create(builder, loc, EncoderType::get(ctx),
                                       function.getArgument(0));
      return encoder;
    }
    if (kind == EvkMapType::getMnemonic()) {
      if (!evkMap) {
        FailureOr<Value> ui =
            supportValue(UserInterfaceType::getMnemonic(), helper);
        if (failed(ui)) return failure();
        evkMap = GetEvkMapOp::create(builder, loc, EvkMapType::get(ctx), *ui);
      }
      return evkMap;
    }
    if (kind == EvalKeyType::getMnemonic()) {
      if (!evalKey) {
        FailureOr<Value> map = supportValue(EvkMapType::getMnemonic(), helper);
        if (failed(map)) return failure();
        Value context;
        for (StringRef contextKind :
             {ContextType::getMnemonic(), BootContextType::getMnemonic()}) {
          auto it = supportIndex.find(contextKind);
          if (it != supportIndex.end())
            context = function.getArgument(it->second);
        }
        if (!context)
          return helper.emitOpError()
                 << "needs a multiplication key, but the facade owns no "
                    "evaluation context";
        evalKey = GetMultKeyOp::create(builder, loc, EvalKeyType::get(ctx),
                                       *map, context);
      }
      return evalKey;
    }
    return helper.emitOpError()
           << "needs a " << kind << " the facade cannot supply";
  }

  ModuleOp module;
  Location loc;
  std::string name;
  StringRef role;
  StringRef entryName;
  Type primaryContext;
  OpBuilder builder;
  SmallVector<StringRef> required;
  SmallVector<DataArgument> dataArguments;
  llvm::StringMap<unsigned> supportIndex;
  func::FuncOp function;
  Value encoder;
  Value evkMap;
  Value evalKey;
};

// The single data argument of a client helper.
FailureOr<Type> singleDataArgument(func::FuncOp helper, unsigned expected) {
  SmallVector<Type> data;
  for (Type type : helper.getArgumentTypes())
    if (supportKind(type).empty()) data.push_back(type);
  if (data.size() != expected)
    return helper.emitOpError()
           << "must take " << expected << " data argument(s), but takes "
           << data.size();
  if (helper.getNumResults() != 1)
    return helper.emitOpError() << "must produce one value";
  return expected ? data.front() : Type();
}

// The element type a setup function's rank-0 tensor result holds: the context
// kind its caller owns.
FailureOr<Type> ownedContext(func::FuncOp setup) {
  if (setup.getNumResults() == 1)
    if (auto tensor = dyn_cast<RankedTensorType>(setup.getResultTypes()[0]))
      if (isContextKind(getSupportKind(tensor.getElementType())))
        return tensor.getElementType();
  return setup.emitOpError("must return the context it creates");
}

// The entry's data arguments, in order: the logical input each is fed from,
// or -1 for an encrypted zero, whose helper index is `zero`.
struct EntryArgument {
  int64_t input;
  int64_t zero;
};

FailureOr<SmallVector<EntryArgument>> entryArguments(func::FuncOp contract) {
  SmallVector<EntryArgument> result;
  int64_t input = 0;
  for (auto [index, type] : llvm::enumerate(contract.getArgumentTypes())) {
    if (!supportKind(type).empty()) continue;
    if (auto zero = contract.getArgAttrOfType<DictionaryAttr>(
            index, kClientEncZeroArgAttrName)) {
      auto zeroIndex =
          dyn_cast_or_null<IntegerAttr>(zero.get(kClientHelperIndex));
      if (!zeroIndex || zeroIndex.getInt() < 0)
        return contract.emitOpError("invalid encrypted-zero index");
      result.push_back({-1, zeroIndex.getInt()});
      continue;
    }
    result.push_back({input++, -1});
  }
  return result;
}

LogicalResult buildEncrypt(ModuleOp module, EntryFunctions& functions,
                           ArrayRef<EntryArgument> arguments,
                           Type clientContext) {
  FacadeBuilder facade(module, functions.contract.getLoc(),
                       functions.entryName + "__encrypt_inputs",
                       kFacadeEncryptRole, functions.entryName, clientContext);
  MLIRContext* ctx = module.getContext();
  // One call per encrypted entry argument; a cleartext scalar that reaches
  // the entry unencrypted has no helper and is not the facade's concern.
  SmallVector<func::FuncOp> helpers;
  for (EntryArgument argument : arguments) {
    func::FuncOp helper =
        argument.zero >= 0
            ? findIndexedHelper(functions.zeroHelpers, argument.zero)
            : findIndexedHelper(functions.inputHelpers, argument.input);
    if (!helper) {
      if (argument.zero >= 0)
        return functions.contract.emitOpError()
               << "is missing the encryption helper for encrypted zero "
               << argument.zero;
      Type type =
          cast<TypeAttr>(getLogicalTypes(functions.contract,
                                         kEntryInputTypes)[argument.input])
              .getValue();
      if (isa<ShapedType>(type)) {
        // A program that is only ever driven through its helpers still
        // lowers; only the generated interface is off the table.
        functions.contract.emitWarning()
            << "entry input " << argument.input
            << " reaches the server as cleartext, which the entry interface "
               "does not support; no encrypt facade is generated";
        return success();
      }
      helpers.push_back({});
      continue;
    }
    FailureOr<Type> data =
        singleDataArgument(helper, argument.zero >= 0 ? 0 : 1);
    if (failed(data)) return failure();
    facade.requireSupportOf(helper);
    if (*data)
      facade.addDataArgument(
          *data, {NamedAttribute(StringAttr::get(ctx, kEntryInputArgAttrName),
                                 IntegerAttr::get(IntegerType::get(ctx, 64),
                                                  argument.input))});
    helpers.push_back(helper);
  }
  return facade.build([&](ValueRange data) -> FailureOr<SmallVector<Value>> {
    SmallVector<Value> results;
    unsigned next = 0;
    for (auto [argument, helper] : llvm::zip(arguments, helpers)) {
      if (!helper) continue;
      FailureOr<ValueRange> result = facade.call(
          helper, argument.zero >= 0 ? ValueRange{} : data.slice(next++, 1));
      if (failed(result)) return failure();
      results.push_back(result->front());
    }
    return results;
  });
}

LogicalResult buildDecrypt(ModuleOp module, EntryFunctions& functions,
                           Type clientContext) {
  FacadeBuilder facade(module, functions.contract.getLoc(),
                       functions.entryName + "__decrypt_outputs",
                       kFacadeDecryptRole, functions.entryName, clientContext);
  unsigned count =
      getLogicalTypes(functions.contract, kEntryResultTypes).size();
  SmallVector<func::FuncOp> helpers;
  for (unsigned index = 0; index < count; ++index) {
    func::FuncOp helper = findIndexedHelper(functions.outputHelpers, index);
    if (!helper) {
      functions.contract.emitWarning()
          << "is missing the decryption helper for result " << index
          << ", which the entry interface needs; no decrypt facade is "
             "generated";
      return success();
    }
    FailureOr<Type> data = singleDataArgument(helper, 1);
    if (failed(data)) return failure();
    facade.requireSupportOf(helper);
    facade.addDataArgument(*data);
    helpers.push_back(helper);
  }
  return facade.build([&](ValueRange data) -> FailureOr<SmallVector<Value>> {
    SmallVector<Value> results;
    for (auto [index, helper] : llvm::enumerate(helpers)) {
      FailureOr<ValueRange> result = facade.call(helper, data.slice(index, 1));
      if (failed(result)) return failure();
      results.push_back(result->front());
    }
    return results;
  });
}

LogicalResult buildEvaluate(ModuleOp module, EntryFunctions& functions,
                            ArrayRef<EntryArgument> arguments,
                            Type serverContext) {
  FacadeBuilder facade(module, functions.evaluate.getLoc(),
                       functions.entryName + "__evaluate", kFacadeEvaluateRole,
                       functions.entryName, serverContext);
  MLIRContext* ctx = module.getContext();
  facade.requireSupportOf(functions.evaluate);
  // The helper's data arguments are the entry's, then the preprocessed values.
  unsigned position = 0;
  for (Type type : functions.evaluate.getArgumentTypes()) {
    if (!supportKind(type).empty()) continue;
    SmallVector<NamedAttribute> attrs;
    if (position < arguments.size()) {
      if (arguments[position].input >= 0)
        attrs.emplace_back(StringAttr::get(ctx, kEntryInputArgAttrName),
                           IntegerAttr::get(IntegerType::get(ctx, 64),
                                            arguments[position].input));
    } else {
      attrs.emplace_back(StringAttr::get(ctx, kPreparedArgAttrName),
                         UnitAttr::get(ctx));
    }
    facade.addDataArgument(type, attrs);
    ++position;
  }
  if (position < arguments.size())
    return functions.evaluate.emitOpError()
           << "takes fewer data arguments than the entry has";
  return facade.build([&](ValueRange data) -> FailureOr<SmallVector<Value>> {
    FailureOr<ValueRange> results = facade.call(functions.evaluate, data);
    if (failed(results)) return failure();
    return SmallVector<Value>(*results);
  });
}

LogicalResult buildPreprocess(ModuleOp module, EntryFunctions& functions,
                              Type serverContext) {
  FacadeBuilder facade(module, functions.preprocess.getLoc(),
                       functions.entryName + "__preprocess",
                       kFacadePreprocessRole, functions.entryName,
                       serverContext);
  for (Type type : functions.preprocess.getArgumentTypes()) {
    if (!supportKind(type).empty()) continue;
    functions.preprocess.emitWarning(
        "forwards a cleartext entry argument, which the entry interface does "
        "not support; no preprocess facade is generated");
    return success();
  }
  facade.requireSupportOf(functions.preprocess);
  return facade.build([&](ValueRange) -> FailureOr<SmallVector<Value>> {
    FailureOr<ValueRange> results = facade.call(functions.preprocess, {});
    if (failed(results)) return failure();
    return SmallVector<Value>(*results);
  });
}

struct CheddarBuildEntryInterface
    : impl::CheddarBuildEntryInterfaceBase<CheddarBuildEntryInterface> {
  using CheddarBuildEntryInterfaceBase::CheddarBuildEntryInterfaceBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    // A program entered below the secret level has no cleartext contract and
    // so no facades; its setup and evaluate helpers stand on their own.
    bool hasContract = false;
    module.walk([&](func::FuncOp function) {
      hasContract |= hasInterfaceRole(function, kEntryRole);
    });
    if (!hasContract) return;
    FailureOr<EntryFunctions> functions =
        findEntryFunctions(module, entryFunction);
    if (failed(functions)) return signalPassFailure();
    if (!functions->setup) {
      module.emitError() << "entry @" << functions->entryName
                         << " has no setup function to take its context from";
      return signalPassFailure();
    }
    if (!getLogicalTypes(functions->contract, kEntryInputTypes) ||
        !getLogicalTypes(functions->contract, kEntryResultTypes)) {
      functions->contract.emitOpError(
          "is missing its cleartext input and result types");
      return signalPassFailure();
    }
    FailureOr<Type> clientContext = ownedContext(functions->setup);
    FailureOr<Type> serverContext = ownedContext(
        functions->serverSetup ? functions->serverSetup : functions->setup);
    FailureOr<SmallVector<EntryArgument>> arguments =
        entryArguments(functions->contract);
    if (failed(clientContext) || failed(serverContext) || failed(arguments))
      return signalPassFailure();
    if (failed(buildEncrypt(module, *functions, *arguments, *clientContext)) ||
        failed(buildDecrypt(module, *functions, *clientContext)) ||
        failed(buildEvaluate(module, *functions, *arguments, *serverContext)) ||
        (functions->preprocess &&
         failed(buildPreprocess(module, *functions, *serverContext))))
      return signalPassFailure();
  }
};

}  // namespace

}  // namespace mlir::heir::cheddar
