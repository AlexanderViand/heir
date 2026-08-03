#include "lib/Conversions/CheddarToEmitC/CheddarToEmitC.h"

#include <cstdio>
#include <functional>
#include <optional>
#include <string>

#include "lib/Dialect/Cheddar/IR/CheddarDialect.h"
#include "lib/Dialect/Cheddar/IR/CheddarOps.h"
#include "lib/Dialect/Cheddar/IR/CheddarTypes.h"
#include "lib/Utils/ConversionUtils.h"
#include "lib/Utils/TargetUtils.h"
#include "llvm/include/llvm/ADT/APFloat.h"          // from @llvm-project
#include "llvm/include/llvm/ADT/DenseSet.h"         // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/SmallString.h"      // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"      // from @llvm-project
#include "llvm/include/llvm/ADT/StringSet.h"        // from @llvm-project
#include "llvm/include/llvm/Support/FileSystem.h"   // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/ConvertToEmitC/ToEmitCInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/MemRefToEmitC/MemRefToEmitC.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/EmitC/IR/EmitC.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/Transforms/FuncConversions.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/SymbolTable.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Interfaces/DestinationStyleOpInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir {

#define GEN_PASS_DEF_CHEDDARTOEMITC
#define GEN_PASS_DEF_CHEDDAREXTERNALIZEWEIGHTS
#include "lib/Conversions/CheddarToEmitC/CheddarToEmitC.h.inc"

namespace {

using ::mlir::emitc::CallOpaqueOp;
using ::mlir::emitc::LValueType;
using ::mlir::emitc::MemberCallOpaqueOp;
using ::mlir::emitc::OpaqueAttr;
using ::mlir::emitc::OpaqueType;
using ::mlir::emitc::PointerType;
using ::mlir::emitc::VariableOp;
using ::mlir::emitc::VerbatimOp;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// The CHEDDAR payload C++ type name for a cheddar element type, or "" if `t`
// isn't a (move-only) cheddar payload type.
std::string payloadTypeName(Type t) {
  if (isa<cheddar::CiphertextType>(t)) return "Ciphertext<word>";
  if (isa<cheddar::PlaintextType>(t)) return "Plaintext<word>";
  if (isa<cheddar::ConstantType>(t)) return "Constant<word>";
  if (isa<cheddar::EvalKeyType>(t)) return "EvaluationKey<word>";
  if (isa<cheddar::LinearTransformType>(t))
    return "std::shared_ptr<cheddar::LinearTransform<word>>";
  return "";
}

// The scalar C++ element type name for a memref element: a cheddar payload, or
// a plain float. "" if `t` is neither (the buffer is left to upstream / fails).
std::string scalarCppName(Type t) {
  std::string p = payloadTypeName(t);
  if (!p.empty()) return p;
  if (auto f = dyn_cast<FloatType>(t)) {
    if (f.getWidth() == 32) return "float";
    if (f.getWidth() == 64) return "double";
  }
  return "";
}

// Build the (nested) `std::array` C++ type for a buffer shape + element name.
// shape [1, 1024], elt "float" -> "std::array<std::array<float, 1024>, 1>".
std::string stdArrayName(ArrayRef<int64_t> shape, StringRef elt) {
  std::string s = elt.str();
  for (int64_t d : llvm::reverse(shape))
    s = "std::array<" + s + ", " + std::to_string(d) + ">";
  return s;
}

// The owning C++ smart-pointer type name for a cheddar setup-handle element
// (context / user_interface), or "" otherwise. A `memref<!cheddar.X>` of one of
// these is an OWNING destination (written by
// create_context/create_user_interface and handed back through a `__configure`
// out-param), as opposed to the bare borrowed handle the compute funcs take
// (`Context<word>*` / `UserInterface<word>*`).
std::string owningHandleTypeName(Type t) {
  if (isa<cheddar::ContextType>(t)) return "std::shared_ptr<Context<word>>";
  if (isa<cheddar::BootContextType>(t))
    return "std::shared_ptr<BootContext<word>>";
  if (isa<cheddar::UserInterfaceType>(t))
    return "std::unique_ptr<UserInterface<word>>";
  return "";
}

// True if an emitc opaque value type names a (move-only) cheddar payload, i.e.
// the buffer's elements are move-only. Used to choose move vs copy semantics.
bool opaqueNamesPayload(StringRef name) {
  return name.contains("Ciphertext<word>") ||
         name.contains("Plaintext<word>") || name.contains("Constant<word>") ||
         name.contains("EvaluationKey<word>") ||
         name.contains("LinearTransform<word>");
}

std::string intLit(IntegerAttr a) { return std::to_string(a.getInt()); }

std::string floatLit(FloatAttr a) {
  // Full double precision (%.17g round-trips exactly); the default formatv
  // precision silently corrupts literal scales.
  char buf[40];
  std::snprintf(buf, sizeof(buf), "%.17g", a.getValueAsDouble());
  return std::string(buf);
}

std::string floatArrayLit(ArrayAttr a) {
  std::string s = "{";
  for (size_t i = 0; i < a.size(); ++i) {
    if (i > 0) s += ", ";
    s += floatLit(cast<FloatAttr>(a[i]));
  }
  return s + "}";
}

std::string placeholders(size_t count) {
  std::string result;
  for (size_t i = 0; i < count; ++i) {
    if (i > 0) result += ", ";
    result += "{}";
  }
  return result;
}

int64_t numElements(ArrayRef<int64_t> shape) {
  int64_t result = 1;
  for (int64_t dim : shape) result *= dim;
  return result;
}

// Flat `<elt>*` to a buffer's first element: the value itself if it is already
// a pointer (e.g. a subview slice), else `&array[0]..[0]`. Emitting this as SSA
// `address_of(subscript(buf, 0..0))` -- not a baked `&buf[0]..[0]` string --
// lets cheddar-emitc-boundary recognize and flatten a result out-param.
Value addressOfFirstElement(OpBuilder& b, Location loc, Value array) {
  if (isa<emitc::PointerType>(array.getType())) return array;
  auto arrayTy = cast<emitc::ArrayType>(array.getType());
  auto sizeT = emitc::SizeTType::get(b.getContext());
  SmallVector<Value> zeroIdxs;
  for (size_t i = 0; i < arrayTy.getShape().size(); ++i)
    zeroIdxs.push_back(emitc::LiteralOp::create(b, loc, sizeT, "0"));
  auto lvalueTy = emitc::LValueType::get(arrayTy.getElementType());
  Value firstElement =
      emitc::SubscriptOp::create(b, loc, lvalueTy, array, zeroIdxs);
  return emitc::AddressOfOp::create(
      b, loc, emitc::PointerType::get(arrayTy.getElementType()), firstElement);
}

// Emit `receiver.method(out, args..., extra)` (or `receiver->method(...)` for a
// pointer receiver). Trailing literal text (`extra`) is appended as one opaque
// constant arg.
void emitOutParamCall(OpBuilder& b, Location loc, Value receiver,
                      StringRef method, Value out, ValueRange args,
                      StringRef extra = "") {
  SmallVector<Value> argOperands{out};
  argOperands.append(args.begin(), args.end());
  ArrayAttr argsAttr;
  if (!extra.empty()) {
    SmallVector<Attribute> a;
    for (size_t i = 0; i < argOperands.size(); ++i)
      a.push_back(b.getIndexAttr(i));
    a.push_back(emitc::OpaqueAttr::get(b.getContext(), extra));
    argsAttr = b.getArrayAttr(a);
  }
  MemberCallOpaqueOp::create(b, loc, /*resultTypes=*/TypeRange{}, receiver,
                             b.getStringAttr(method), argsAttr,
                             /*template_args=*/ArrayAttr{}, argOperands);
}

//===----------------------------------------------------------------------===//
// Type conversions
//===----------------------------------------------------------------------===//

// Add the cheddar-specific conversions to the shared EmitC type converter:
// cheddar handle/payload types and payload buffers. A move-only payload buffer
// (`memref<!cheddar.X>`) maps to an `emitc.lvalue` of the payload type for a
// scalar (rank-0) buffer, or to an `emitc.array` for a rank-1 buffer (looped
// kernels). Function-boundary buffers are re-typed to C++ references by the
// `cheddar-emitc-boundary` pass (a func cannot carry lvalue args).
void addCheddarEmitCTypeConversions(TypeConverter& tc, MLIRContext* ctx) {
  // Identity for the emitc types we produce, so they are recognised as already
  // legal (the shared EmitCTypeConverter's generic rule rejects e.g. lvalue,
  // leaving a converted func signature perpetually "illegal").
  tc.addConversion([](emitc::LValueType t) -> Type { return t; });
  tc.addConversion([](emitc::PointerType t) -> Type { return t; });
  tc.addConversion([](emitc::ArrayType t) -> Type { return t; });
  tc.addConversion([](emitc::OpaqueType t) -> Type { return t; });
  tc.addConversion([](emitc::SizeTType t) -> Type { return t; });
  tc.addConversion([ctx](cheddar::ParameterType) -> Type {
    return OpaqueType::get(ctx, "Parameter<word>");
  });
  tc.addConversion([ctx](cheddar::ContextType) -> Type {
    return PointerType::get(ctx, OpaqueType::get(ctx, "Context<word>"));
  });
  tc.addConversion([ctx](cheddar::BootContextType) -> Type {
    return PointerType::get(ctx, OpaqueType::get(ctx, "BootContext<word>"));
  });
  tc.addConversion([ctx](cheddar::UserInterfaceType) -> Type {
    return PointerType::get(ctx, OpaqueType::get(ctx, "UserInterface<word>"));
  });
  tc.addConversion([ctx](cheddar::EncoderType) -> Type {
    return OpaqueType::get(ctx, "Encoder<word>");
  });
  tc.addConversion([ctx](cheddar::EvkMapType) -> Type {
    return OpaqueType::get(ctx, "EvkMap<word>");
  });
  tc.addConversion([ctx](cheddar::EvalKeyType) -> Type {
    return OpaqueType::get(ctx, "EvaluationKey<word>");
  });
  tc.addConversion([ctx](cheddar::CiphertextType) -> Type {
    return OpaqueType::get(ctx, "Ciphertext<word>");
  });
  tc.addConversion([ctx](cheddar::PlaintextType) -> Type {
    return OpaqueType::get(ctx, "Plaintext<word>");
  });
  tc.addConversion([ctx](cheddar::ConstantType) -> Type {
    return OpaqueType::get(ctx, "Constant<word>");
  });
  tc.addConversion([ctx](cheddar::LinearTransformType) -> Type {
    return OpaqueType::get(ctx,
                           "std::shared_ptr<cheddar::LinearTransform<word>>");
  });
  tc.addConversion(
      [ctx](IndexType) -> Type { return emitc::SizeTType::get(ctx); });
  // Bufferized buffers split by element kind:
  //  * move-only cheddar PAYLOAD (ciphertext/plaintext/constant/key): a place
  //    carried as `lvalue<opaque>` -- rank-0 -> `T name;`; rank>=1 -> a
  //    (nested) `std::array` (`std::array<T,N> name;`). std::array (not a C
  //    array) so it binds to the `std::array<T,N>&` function-boundary refs and
  //    the harness; subscripting works via the patched emitc.subscript
  //    (lvalue<opaque> base).
  //  * FLOAT message/weight buffers: a C array (`emitc.array`). emitc.global
  //    (weight constants) requires an emitc.array type and emitc subscripts C
  //    arrays natively, so floats stay C arrays; a strided subview slice
  //    feeding cheddar.encode becomes a raw pointer.
  // Layout is otherwise ignored (a payload subview slice is contiguous, so its
  // shape alone names the std::array). Other memrefs are left to upstream.
  tc.addConversion([ctx](MemRefType type) -> std::optional<Type> {
    Type eltType = type.getElementType();
    // Owning setup-handle destination (rank-0 context/user_interface buffer):
    // an `lvalue` of the owning smart-pointer type, re-typed to `T&` at the
    // function boundary so `__configure` writes the handle back to the caller.
    if (type.getRank() == 0) {
      std::string owning = owningHandleTypeName(eltType);
      if (!owning.empty())
        return Type(LValueType::get(OpaqueType::get(ctx, owning)));
    }
    std::string elt = scalarCppName(eltType);
    if (elt.empty()) return std::nullopt;
    bool payload = !payloadTypeName(eltType).empty();
    if (type.getRank() == 0)
      return Type(LValueType::get(OpaqueType::get(ctx, elt)));
    if (payload) {
      if (!type.hasStaticShape() || llvm::is_contained(type.getShape(), 0))
        return Type();
      return Type(LValueType::get(
          OpaqueType::get(ctx, stdArrayName(type.getShape(), elt))));
    }
    // Float: a strided subview slice -> raw pointer; otherwise a C array.
    if (isa<StridedLayoutAttr>(type.getLayout()))
      return Type(emitc::PointerType::get(eltType));
    if (!type.hasStaticShape() || llvm::is_contained(type.getShape(), 0))
      return Type();
    return Type(emitc::ArrayType::get(type.getShape(), eltType));
  });
}

// emitc::OpaqueType doesn't implement MemRefElementTypeInterface upstream;
// needed if `memref<Nx!emitc.opaque<...>>` is ever formed. Marker-only.
struct EmitCOpaqueAsMemRefElement
    : public mlir::MemRefElementTypeInterface::ExternalModel<
          EmitCOpaqueAsMemRefElement, mlir::emitc::OpaqueType> {};

// The getter-style setup ops hand back a const& to a move-only / non-assignable
// value; a value-materialising lowering can't compile them. Reject up front.
bool diagnoseUnsupportedGetters(Operation* root) {
  bool found = false;
  root->walk([&](Operation* op) {
    if (isa<cheddar::GetEvkMapOp, cheddar::GetMultKeyOp, cheddar::GetEncoderOp>(
            op)) {
      op->emitError()
          << "cheddar-to-emitc: lowering of '" << op->getName().getStringRef()
          << "' is not supported: it returns a const reference to a "
             "move-only/non-assignable value. Pass the "
             "key/map/encoder as a function argument, or look it up "
             "inline at the use site.";
      found = true;
    }
  });
  return found;
}

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

// Generic destination-passing op -> a single out-parameter method call:
//   receiver->Method(dest, inputs..., extra)
template <typename Op>
struct OutParamDpsPattern : public OpConversionPattern<Op> {
  OutParamDpsPattern(const TypeConverter& tc, MLIRContext* ctx,
                     StringRef method,
                     std::function<std::string(Op)> extra = nullptr)
      : OpConversionPattern<Op>(tc, ctx),
        method(method.str()),
        extra(std::move(extra)) {}

  LogicalResult matchAndRewrite(
      Op op, typename Op::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto dpsOp = cast<DestinationStyleOpInterface>(op.getOperation());
    unsigned initIdx = dpsOp.getDpsInitOperand(0)->getOperandNumber();
    auto operands = adaptor.getOperands();
    Value receiver = operands[0];
    Value dest = operands[initIdx];
    SmallVector<Value> inputs;
    for (unsigned i = 1; i < operands.size(); ++i)
      if (i != initIdx) inputs.push_back(operands[i]);
    std::string extraStr = extra ? extra(op) : "";
    emitOutParamCall(rewriter, op.getLoc(), receiver, method, dest, inputs,
                     extraStr);
    rewriter.eraseOp(op);
    return success();
  }

  std::string method;
  std::function<std::string(Op)> extra;
};

// A setup op that writes an owning handle into its destination buffer:
//   create_context %params, %out          -> out =
//   Context<word>::Create(params); create_user_interface %ctx, %out      -> out
//   = make_unique<UI<word>>(ctx);
// Operands are (source, output); the RHS callee is the only thing that differs.
template <typename Op>
struct ConvertSetupAssign : public OpConversionPattern<Op> {
  ConvertSetupAssign(const TypeConverter& tc, MLIRContext* ctx,
                     StringRef rhsCallee)
      : OpConversionPattern<Op>(tc, ctx), rhsCallee(rhsCallee.str()) {}

  LogicalResult matchAndRewrite(
      Op op, typename Op::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto operands = adaptor.getOperands();  // [source, output]
    VerbatimOp::create(rewriter, op.getLoc(), "{} = " + rhsCallee + "({});",
                       ValueRange{operands[1], operands[0]});
    rewriter.eraseOp(op);
    return success();
  }

  std::string rhsCallee;
};

// MakeParameter: construct a `cheddar::Parameter<word>` value from the CKKS
// modulus chain carried as attributes:
//   Parameter<word>(logN, scale, maxLevel, {{1,0},...}, {Q...}, {P...})
struct ConvertMakeParameter
    : public OpConversionPattern<cheddar::MakeParameterOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::MakeParameterOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Type t = typeConverter->convertType(op.getResult().getType());
    int64_t logN = op.getLogN().getInt();
    int64_t logScale = op.getLogScale().getInt();
    ArrayRef<int64_t> mainPrimes = op.getMainPrimes();
    ArrayRef<int64_t> auxPrimes = op.getAuxPrimes();
    // Canonical one-main-prime-per-level layout: default encryption level is
    // the deepest level, #mainPrimes - 1, unless overridden (bootstrapping
    // pins it below the chain top so the boot-circuit primes sit above it).
    int64_t maxLevel = static_cast<int64_t>(mainPrimes.size()) - 1;
    int64_t defaultEncLevel = op.getDefaultEncryptionLevel()
                                  ? op.getDefaultEncryptionLevelAttr().getInt()
                                  : maxLevel;

    // level_config: the canonical one-main-prime-per-level layout
    // {{1, 0}, {2, 0}, ..., {#Q, 0}}.
    std::string levelCfg = "std::vector<std::pair<int, int>>{";
    for (size_t i = 0; i < mainPrimes.size(); ++i) {
      if (i) levelCfg += ", ";
      levelCfg += "{" + std::to_string(i + 1) + ", 0}";
    }
    levelCfg += "}";

    // Primes are stored as i64 attrs but represent uint64 CKKS moduli.
    auto primeVec = [](ArrayRef<int64_t> primes) {
      std::string s = "std::vector<word>{";
      for (size_t i = 0; i < primes.size(); ++i) {
        if (i) s += ", ";
        s += std::to_string(static_cast<uint64_t>(primes[i])) + "ULL";
      }
      s += "}";
      return s;
    };

    std::string scaleLit = "static_cast<double>(static_cast<word>(1) << " +
                           std::to_string(logScale) + ")";

    // CHEDDAR's Context stores a `const Parameter<word>&` (it does NOT copy),
    // so the Parameter must outlive the Context the generated `__configure`
    // hands back. Emit it as a function-local `static` (constructed once on
    // first call) and return an `emitc.literal` reference to it, rather than a
    // plain local that would dangle once `__configure` returns.
    std::string ctorArgs = std::to_string(logN) + ", " + scaleLit + ", " +
                           std::to_string(defaultEncLevel) + ", " + levelCfg +
                           ", " + primeVec(mainPrimes) + ", " +
                           primeVec(auxPrimes);
    StringRef name = "cheddar_param";
    VerbatimOp::create(rewriter, op.getLoc(),
                       ("static Parameter<word> " + name +
                        " = Parameter<word>(" + ctorArgs + ");")
                           .str(),
                       ValueRange{});
    // Bootstrapping needs the secret-key hamming weights set on the Parameter
    // (sparse-secret bootstrapping); CHEDDAR's BootContext relies on them.
    if (op.getDenseHammingWeight())
      VerbatimOp::create(
          rewriter, op.getLoc(),
          (name + ".SetDenseHammingWeight(" +
           std::to_string(op.getDenseHammingWeightAttr().getInt()) + ");")
              .str(),
          ValueRange{});
    if (op.getSparseHammingWeight())
      VerbatimOp::create(
          rewriter, op.getLoc(),
          (name + ".SetSparseHammingWeight(" +
           std::to_string(op.getSparseHammingWeightAttr().getInt()) + ");")
              .str(),
          ValueRange{});
    auto lit = emitc::LiteralOp::create(rewriter, op.getLoc(), t, name);
    rewriter.replaceOp(op, lit.getResult());
    return success();
  }
};

// Recover the Context a UserInterface was created from by walking back the
// destination-passing `$ui` chain (prepare_rot_key / prepare_bootstrap thread
// it through) to the create_user_interface that owns it.
// cheddar.prepare_rot_key carries no ctx operand, but cyclops' post-CYC-173
// keygen needs a SecretId, which only the Context's Parameter can supply -- so
// the emitted call takes the context and the shim SFINAEs on whether the fork
// wants it.
static Value findContextForUi(Value ui) {
  while (ui) {
    Operation* def = ui.getDefiningOp();
    if (!def) return nullptr;
    if (auto cui = dyn_cast<cheddar::CreateUserInterfaceOp>(def))
      return cui.getCtx();
    if (auto prk = dyn_cast<cheddar::PrepareRotKeyOp>(def)) {
      ui = prk.getUi();
      continue;
    }
    if (auto pb = dyn_cast<cheddar::PrepareBootstrapOp>(def)) {
      ui = pb.getUi();
      continue;
    }
    return nullptr;
  }
  return nullptr;
}

// After bufferization the DPS `$ui` is a memref that create_user_interface
// *writes into* rather than defines, so the def-chain walk above dead-ends.
// Two fallbacks, in order: the create_user_interface in this function whose
// destination is that same memref, then the enclosing function's context
// parameter (in a generated __configure the context is an out-param).
static Value findContextFallback(Operation* op, Value ui) {
  auto fn = op->getParentOfType<func::FuncOp>();
  if (!fn) return nullptr;
  Value found;
  fn.walk([&](cheddar::CreateUserInterfaceOp cui) {
    if (found) return;
    for (Value operand : cui->getOperands())
      if (operand == ui) found = cui.getCtx();
  });
  if (found) return found;
  auto mentionsContext = [](Type t) {
    std::string s;
    llvm::raw_string_ostream os(s);
    t.print(os);
    if (StringRef(s).contains_insensitive("boot")) return false;
    return StringRef(s).contains("cheddar.context") ||
           StringRef(s).contains("Context<");
  };
  for (Value arg : fn.getArguments())
    if (mentionsContext(arg.getType())) return arg;
  return nullptr;
}

struct ConvertPrepareRotKey
    : public OpConversionPattern<cheddar::PrepareRotKeyOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::PrepareRotKeyOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    std::string extra =
        intLit(op.getDistanceAttr()) + ", " + intLit(op.getMaxLevelAttr());
    Value origCtx = findContextForUi(op.getUi());
    if (!origCtx) origCtx = findContextFallback(op, op.getUi());
    if (!origCtx)
      return rewriter.notifyMatchFailure(
          op,
          "cannot locate the Context owning this $ui, so cyclops' SecretId "
          "is unavailable");
    Value ctxV = rewriter.getRemappedValue(origCtx);
    if (!ctxV) ctxV = origCtx;
    // Linear-transform rotation keys go through the fork-dispatching wrapper
    // (level-specific on cyclops, chain-max + dedupe on scale-snu cheddar).
    std::string call = op.getChainMaxLevelAttr()
                           ? "PrepareLintransRotKey({}, " + extra + ", " +
                                 intLit(op.getChainMaxLevelAttr()) + ", {});"
                           : "HeirPrepareRotKey({}, " + extra + ", {});";
    VerbatimOp::create(rewriter, op.getLoc(), call,
                       ValueRange{adaptor.getUi(), ctxV});
    rewriter.eraseOp(op);
    return success();
  }
};

// create_boot_context: build the owning BootContext from the Parameter plus the
// CtS/StC level budgets (BootParameter). Operands are (params, output).
//   {out} = BootContext<word>::Create(param,
//               BootParameter(param.max_level_, numCts, numStc));
struct ConvertCreateBootContext
    : public OpConversionPattern<cheddar::CreateBootContextOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::CreateBootContextOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    std::string numCts = std::to_string(op.getNumCtsLevels().getInt());
    std::string numStc = std::to_string(op.getNumStcLevels().getInt());
    // log_message_ratio (4th BootParameter arg) governs EvalMod precision; when
    // unset, omit it so CHEDDAR's BootParameter default applies.
    std::string ratioArg;
    if (auto r = op.getLogMessageRatioAttr())
      ratioArg = ", " + std::to_string(r.getInt());
    VerbatimOp::create(
        rewriter, op.getLoc(),
        "{} = BootContext<word>::Create({}, BootParameter({}.max_level_, " +
            numCts + ", " + numStc + ratioArg + "));",
        ValueRange{adaptor.getOutput(), adaptor.getParams(),
                   adaptor.getParams()});
    rewriter.eraseOp(op);
    return success();
  }
};

// prepare_bootstrap: the one-time bootstrap precompute + rotation-key request.
// Mutates the boot context (eval-mod / special-FFT precompute) and threads the
// required rotation keys into the user interface. Mirrors CHEDDAR's canonical
// boot preparation sequence.
struct ConvertPrepareBootstrap
    : public OpConversionPattern<cheddar::PrepareBootstrapOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::PrepareBootstrapOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    std::string n = std::to_string(op.getNumSlots().getInt());
    Value ctx = adaptor.getCtx();
    Value ui = adaptor.getUi();
    VerbatimOp::create(rewriter, op.getLoc(), "{}->PrepareEvalMod();",
                       ValueRange{ctx});
    // kImaginaryRemoving: CKKS bootstrap (CoeffToSlot/EvalMod/SlotToCoeff)
    // leaves the EvalMod error + conjugate term in the imaginary slots; the
    // default kNormal variant does NOT strip them. A real-valued pipeline must
    // remove them at prepare time (the variant also halves the SlotToCoeff
    // constant, so bolting on a post-hoc HConjAdd would double the real part
    // instead). Left as kNormal, a boot output carries a hidden imaginary
    // component that the downstream (complex-linear) layers propagate and the
    // Chebyshev eval_poly -- bounded only on the real axis -- detonates on
    // (~1e16+).
    VerbatimOp::create(rewriter, op.getLoc(),
                       "{}->PrepareEvalSpecialFFT(" + n +
                           ", cheddar::BootVariant::kImaginaryRemoving);",
                       ValueRange{ctx});
    VerbatimOp::create(rewriter, op.getLoc(), "EvkRequest boot_evk_req;",
                       ValueRange{});
    VerbatimOp::create(rewriter, op.getLoc(),
                       "{}->AddRequiredRotations(boot_evk_req, " + n + ");",
                       ValueRange{ctx});
    // Boot keys are built under the boot secret on cyclops (post-CYC-173);
    // scale-snu CHEDDAR takes the request alone. The shim dispatches on which
    // signature exists, so this stays fork-agnostic.
    VerbatimOp::create(rewriter, op.getLoc(),
                       "HeirPrepareBootRotKeys({}, boot_evk_req, {});",
                       ValueRange{ui, ctx});
    // Bootstrap CtS/StC precompute is the largest transient GPU allocation and
    // some forks' binning allocators cache the freed scratch. Reclaim it at
    // this boundary (before weight encode + transform keygen) via a
    // consumer-provided hook: the cyclops prelude calls
    // cheddar::GPU::ReleaseUnusedMemory(); the scale-snu prelude defines it as
    // a no-op (no such pool API). Keeps this emission backend-agnostic, like
    // heir_load_f32 / __heir_debug.
    VerbatimOp::create(rewriter, op.getLoc(), "__medusa_reclaim();",
                       ValueRange{});
    rewriter.eraseOp(op);
    return success();
  }
};

// cheddar.encode: fill a std::vector<Complex> from the float message buffer,
// then `encoder.Encode(out, level, param_.GetScale(level), msg)`.
struct ConvertEncode : public OpConversionPattern<cheddar::EncodeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::EncodeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value out = adaptor.getOutput();
    std::string lvl = std::to_string(op.getLevelAttr().getInt());
    Value msg = adaptor.getMessage();
    int64_t n = 1;
    if (auto sh = dyn_cast<ShapedType>(op.getMessage().getType()))
      for (int64_t d : sh.getShape()) n *= d;
    // The message is a float buffer: a raw pointer (subview slice) -> `{}`, or
    // a C array (whole buffer) -> `&{}[0]`.
    std::string begin =
        isa<emitc::PointerType>(msg.getType()) ? "{}" : "&{}[0]";
    Value vec =
        VariableOp::create(rewriter, op.getLoc(),
                           LValueType::get(OpaqueType::get(
                               rewriter.getContext(), "std::vector<Complex>")),
                           OpaqueAttr::get(rewriter.getContext(), ""));
    VerbatimOp::create(rewriter, op.getLoc(),
                       "{} = std::vector<Complex>(" + begin + ", " + begin +
                           " + " + std::to_string(n) + ");",
                       ValueRange{vec, msg, msg});
    VerbatimOp::create(
        rewriter, op.getLoc(),
        "{}.Encode({}, " + lvl + ", {}.GetScale(" + lvl + "), {});",
        ValueRange{adaptor.getEncoder(), out, adaptor.getEncoder(), vec});
    rewriter.eraseOp(op);
    return success();
  }
};

// cheddar.encode_constant: encoder.EncodeConstant(out, level, scale, number).
struct ConvertEncodeConstant
    : public OpConversionPattern<cheddar::EncodeConstantOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::EncodeConstantOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    std::string lvl = intLit(op.getLevelAttr());
    std::string fmt =
        "{}.EncodeConstant({}, " + lvl + ", {}.GetScale(" + lvl + "), {});";
    VerbatimOp::create(rewriter, op.getLoc(), fmt,
                       ValueRange{adaptor.getEncoder(), adaptor.getOutput(),
                                  adaptor.getEncoder(), adaptor.getValue()});
    rewriter.eraseOp(op);
    return success();
  }
};

// cheddar.decode: decode into a temporary complex vector, copy real parts into
// the float destination buffer.
struct ConvertDecode : public OpConversionPattern<cheddar::DecodeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::DecodeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value dst = adaptor.getValue();
    if (!isa<emitc::ArrayType, emitc::LValueType>(dst.getType()))
      return failure();
    auto memTy = dyn_cast<MemRefType>(op.getValue().getType());
    if (!memTy || !isa<FloatType>(memTy.getElementType())) return failure();
    auto shape = memTy.getShape();
    std::string idxPrefix;
    for (size_t i = 0; i + 1 < shape.size(); ++i) {
      if (shape[i] != 1) return failure();
      idxPrefix += "[0]";
    }
    auto* ctx = rewriter.getContext();
    Value vec = VariableOp::create(
        rewriter, op.getLoc(),
        LValueType::get(OpaqueType::get(ctx, "std::vector<Complex>")),
        OpaqueAttr::get(ctx, ""));
    VerbatimOp::create(
        rewriter, op.getLoc(), "{}.Decode({}, {});",
        ValueRange{adaptor.getEncoder(), vec, adaptor.getPlaintext()});
    VerbatimOp::create(rewriter, op.getLoc(),
                       "for (size_t _i = 0; _i < " +
                           std::to_string(shape.back()) + "; ++_i) {}" +
                           idxPrefix + "[_i] = {}[_i].real();",
                       ValueRange{dst, vec});
    rewriter.eraseOp(op);
    return success();
  }
};

// HRot/HRotAdd/HConj/HConjAdd: look up the rotation/conjugation key inline.
struct ConvertHRot : public OpConversionPattern<cheddar::HRotOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::HRotOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value ui = adaptor.getUi();
    Value out = adaptor.getOutput();
    if (auto sd = op.getStaticDistanceAttr()) {
      std::string d = intLit(sd);
      VerbatimOp::create(
          rewriter, op.getLoc(),
          "{}->HRot({}, {}, HeirRotationKey({}, " + d + ", {}), " + d + ");",
          ValueRange{adaptor.getCtx(), out, adaptor.getInput(), ui,
                     adaptor.getInput()});
    } else {
      Value dyn = adaptor.getDynamicDistance();
      VerbatimOp::create(rewriter, op.getLoc(),
                         "{}->HRot({}, {}, HeirRotationKey({}, {}, {}), {});",
                         ValueRange{adaptor.getCtx(), out, adaptor.getInput(),
                                    ui, dyn, adaptor.getInput(), dyn});
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertHRotAdd : public OpConversionPattern<cheddar::HRotAddOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::HRotAddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value ui = adaptor.getUi();
    std::string d = intLit(op.getDistanceAttr());
    VerbatimOp::create(
        rewriter, op.getLoc(),
        "{}->HRotAdd({}, {}, {}, HeirRotationKey({}, " + d + ", {}), " + d +
            ");",
        ValueRange{adaptor.getCtx(), adaptor.getOutput(), adaptor.getInput(),
                   adaptor.getAddend(), ui, adaptor.getInput()});
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertHConj : public OpConversionPattern<cheddar::HConjOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::HConjOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value ui = adaptor.getUi();
    VerbatimOp::create(rewriter, op.getLoc(),
                       "{}->HConj({}, {}, HeirConjugationKey({}, {}));",
                       ValueRange{adaptor.getCtx(), adaptor.getOutput(),
                                  adaptor.getInput(), ui, adaptor.getInput()});
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertHConjAdd : public OpConversionPattern<cheddar::HConjAddOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::HConjAddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value ui = adaptor.getUi();
    VerbatimOp::create(
        rewriter, op.getLoc(),
        "{}->HConjAdd({}, {}, {}, HeirConjugationKey({}, {}));",
        ValueRange{adaptor.getCtx(), adaptor.getOutput(), adaptor.getInput(),
                   adaptor.getAddend(), ui, adaptor.getInput()});
    rewriter.eraseOp(op);
    return success();
  }
};

// cheddar.linear_transform -> RunLinearTransform<W, word>(out, ctx, in,
// evk_map, diagonals, {indices}, level, bs, gs).
struct ConvertLinearTransform
    : public OpConversionPattern<cheddar::LinearTransformOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::LinearTransformOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto diagTy = cast<ShapedType>(op.getDiagonals().getType());
    int64_t width = diagTy.getShape()[1];
    std::string idxList;
    auto idxAttr = op.getDiagonalIndicesAttr();
    for (size_t i = 0; i < idxAttr.size(); ++i) {
      if (i > 0) idxList += ", ";
      idxList += std::to_string(idxAttr[i]);
    }
    int64_t bs = op.getBsAttr().getInt();
    int64_t gs = op.getGsAttr().getInt();
    SmallVector<Value> operands{adaptor.getOutput(), adaptor.getCtx(),
                                adaptor.getInput(), adaptor.getEvkMap(),
                                adaptor.getDiagonals()};
    std::string trailing = "{" + idxList + "}, " + intLit(op.getLevelAttr()) +
                           ", " + std::to_string(bs) + ", " +
                           std::to_string(gs);
    SmallVector<Attribute> args;
    for (size_t i = 0; i < operands.size(); ++i)
      args.push_back(rewriter.getIndexAttr(i));
    args.push_back(emitc::OpaqueAttr::get(rewriter.getContext(), trailing));
    SmallVector<Attribute> templateArgs{
        emitc::OpaqueAttr::get(rewriter.getContext(), std::to_string(width)),
        emitc::OpaqueAttr::get(rewriter.getContext(), "word")};
    CallOpaqueOp::create(rewriter, op.getLoc(), TypeRange{},
                         rewriter.getStringAttr("RunLinearTransform"), operands,
                         rewriter.getArrayAttr(args),
                         rewriter.getArrayAttr(templateArgs));
    rewriter.eraseOp(op);
    return success();
  }
};

// cheddar.prepare_linear_transform ->
// PrepareLinearTransform<W, word>(out, ctx, diagonals, {indices}, level, bs,
// gs). This call lives in the split __preprocessing function.
struct ConvertPrepareLinearTransform
    : public OpConversionPattern<cheddar::PrepareLinearTransformOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::PrepareLinearTransformOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto diagTy = cast<ShapedType>(op.getDiagonals().getType());
    int64_t width = diagTy.getShape()[1];
    std::string idxList;
    for (auto [i, idx] :
         llvm::enumerate(op.getDiagonalIndicesAttr().asArrayRef())) {
      if (i > 0) idxList += ", ";
      idxList += std::to_string(idx);
    }
    SmallVector<Value> operands{adaptor.getOutput(), adaptor.getCtx(),
                                adaptor.getDiagonals()};
    std::string trailing = "{" + idxList + "}, " + intLit(op.getLevelAttr()) +
                           ", " + intLit(op.getBsAttr()) + ", " +
                           intLit(op.getGsAttr());
    SmallVector<Attribute> args;
    for (size_t i = 0; i < operands.size(); ++i)
      args.push_back(rewriter.getIndexAttr(i));
    args.push_back(emitc::OpaqueAttr::get(rewriter.getContext(), trailing));
    SmallVector<Attribute> templateArgs{
        emitc::OpaqueAttr::get(rewriter.getContext(), std::to_string(width)),
        emitc::OpaqueAttr::get(rewriter.getContext(), "word")};
    CallOpaqueOp::create(rewriter, op.getLoc(), TypeRange{},
                         rewriter.getStringAttr("PrepareLinearTransform"),
                         operands, rewriter.getArrayAttr(args),
                         rewriter.getArrayAttr(templateArgs));
    rewriter.eraseOp(op);
    return success();
  }
};

// cheddar.apply_prepared_linear_transform -> RunPreparedLinearTransform.
struct ConvertApplyPreparedLinearTransform
    : public OpConversionPattern<cheddar::ApplyPreparedLinearTransformOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::ApplyPreparedLinearTransformOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    SmallVector<Value> operands{adaptor.getOutput(), adaptor.getCtx(),
                                adaptor.getInput(), adaptor.getEvkMap(),
                                adaptor.getTransform()};
    SmallVector<Attribute> args;
    for (size_t i = 0; i < operands.size(); ++i)
      args.push_back(rewriter.getIndexAttr(i));
    CallOpaqueOp::create(rewriter, op.getLoc(), TypeRange{},
                         rewriter.getStringAttr("RunPreparedLinearTransform"),
                         operands, rewriter.getArrayAttr(args),
                         /*template_args=*/ArrayAttr{});
    rewriter.eraseOp(op);
    return success();
  }
};

// cheddar.eval_poly -> the real cheddar::EvalPoly<word> class (no free
// `RunEvalPoly` exists). It is a stateful object: construct, Compile, then
// Evaluate. Critically, EvalPoly's Chebyshev basis recurrence (T_{2n}=2T_n^2-1)
// propagates scale as `x_scale*y_scale/param_.GetRescalePrimeProd(level)`, and
// the constructor's `input_scale`/`target_scale`/`input_level` SEED that
// recurrence -- cheddar's asserts only check self-consistency with these args,
// not against the real q-products, so a wrong seed silently squares into a
// catastrophic blow-up (e.g. a degree-15 sign poly returning ~1e105) while the
// scale *label* stays canonical. So we must mirror exactly how cheddar's own
// EvalMod seeds EvalPoly: take the level/scale from the *actual* input
// ciphertext and derive target_scale by the same square/divide recurrence.
//
//   {
//     ConstContextPtr<word> cp(ConstContextPtr<word>(), ctx);
//     int lvl = ctx->param_.NPToLevel(in.GetNP());
//     double is = in.GetScale();
//     double ts = is;
//     ts = ts*ts / ctx->param_.GetRescalePrimeProd(lvl - 0);   // x
//     level_consumption
//     ...
//     cheddar::EvalPoly<word> ep({coeffs}, lvl, is, ts, /*chebyshev=*/true);
//     ep.Compile(cp);
//     ep.Evaluate(cp, out, in, evk.GetMultiplicationKey());
//   }
//
// This needs `in.GetScale()`/`in.GetNP()` (whose receiver is a move-only lvalue
// ciphertext that emitc.member_call_opaque rejects), `ctx->param_` (a nested
// member access), and a per-level recurrence -- so it is emitted as verbatim
// statements. These are all real cheddar public API (the inline equivalent of
// EvalMod), not a shim. A `{ }` block scope destroys the EvalPoly (which holds
// the GPU power basis) right after Evaluate; its locals stay block-local so
// repeated eval_poly ops in one function do not collide.
struct ConvertEvalPoly : public OpConversionPattern<cheddar::EvalPolyOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::EvalPolyOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    Value ctxV = adaptor.getCtx();
    Value in = adaptor.getInput();
    Value out = adaptor.getOutput();
    Value evk = adaptor.getEvkMap();
    // EvalPoly internally drops level_consumption = Log2Ceil(degree+1) levels;
    // the level analysis encodes that as level - outputLevel.
    int64_t levelConsumption =
        op.getLevelAttr().getInt() - op.getOutputLevelAttr().getInt();

    auto emit = [&](const Twine& fmt, ValueRange operands) {
      VerbatimOp::create(rewriter, loc, rewriter.getStringAttr(fmt.str()),
                         operands);
    };

    emit("{", {});
    // Compile/Evaluate take a ConstContextPtr (shared_ptr<const Context>); wrap
    // the raw Context* in a non-owning alias (it does not own the context).
    emit("ConstContextPtr<word> _ep_cp(ConstContextPtr<word>(), {});", {ctxV});
    // level + input scale taken from the actual input ciphertext.
    emit("int _ep_lvl = {}->param_.NPToLevel({}.GetNP());", {ctxV, in});
    emit("double _ep_is = {}.GetScale();", {in});
    // target_scale recurrence: ts <- ts*ts / GetRescalePrimeProd(lvl - i).
    emit("double _ep_ts = _ep_is;", {});
    for (int64_t i = 0; i < levelConsumption; ++i)
      emit(
          "_ep_ts = _ep_ts * _ep_ts / {}->param_.GetRescalePrimeProd(_ep_lvl "
          "- " +
              Twine(i) + ");",
          {ctxV});
    // Construct (no operands -> the coefficient brace-list is emitted
    // verbatim).
    emit("cheddar::EvalPoly<word> _ep(" +
             floatArrayLit(op.getCoefficientsAttr()) +
             ", _ep_lvl, _ep_is, _ep_ts, true);",
         {});
    emit("_ep.Compile(_ep_cp);", {});
    emit("_ep.Evaluate(_ep_cp, {}, {}, HeirMultiplicationKey({}, {}));",
         {out, in, evk, in});
    emit("}", {});

    rewriter.eraseOp(op);
    return success();
  }
};

// A `__heir_debug_*` call (from --lwe-add-debug-port, re-shaped by LWEToCheddar
// to (Encoder, UserInterface, Ciphertext)) -> a free C++ call to an
// externally-defined `__heir_debug(encoder, ui, ct, "name", "metadata")`. The
// debug name/metadata travel as `debug.name`/`debug.metadata` dialect attrs;
// they are baked into the call as trailing string-literal args. Medusa's C++
// prelude defines `__heir_debug` (decrypt + decode + print); the external
// `func.func` declaration is erased by the cheddar-emitc-boundary pass (the
// upstream Cpp emitter cannot print an external func.func declaration).
struct ConvertDebugCall : public OpConversionPattern<func::CallOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      func::CallOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    if (!isDebugPort(op.getCallee())) return failure();
    auto* ctx = rewriter.getContext();
    auto escape = [](StringRef s) {
      std::string out = "\"";
      for (char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        out += c;
      }
      out += '"';
      return out;
    };
    std::string name;
    if (auto n = op->getAttrOfType<StringAttr>("debug.name"))
      name = escape(n.getValue());
    else
      name = "\"\"";
    std::string metadata;
    if (auto m = op->getAttrOfType<StringAttr>("debug.metadata"))
      metadata = escape(m.getValue());
    else
      metadata = "\"\"";
    SmallVector<Value> operands(adaptor.getOperands().begin(),
                                adaptor.getOperands().end());
    SmallVector<Attribute> args;
    for (size_t i = 0; i < operands.size(); ++i)
      args.push_back(rewriter.getIndexAttr(i));
    args.push_back(emitc::OpaqueAttr::get(ctx, name));
    args.push_back(emitc::OpaqueAttr::get(ctx, metadata));
    CallOpaqueOp::create(rewriter, op.getLoc(), TypeRange{},
                         rewriter.getStringAttr("__heir_debug"), operands,
                         rewriter.getArrayAttr(args),
                         /*template_args=*/ArrayAttr{});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// memref op patterns (payload + float)
//===----------------------------------------------------------------------===//

// memref.alloc of a cheddar buffer (payload or float, scalar or std::array) ->
// a stack `emitc.variable` of the converted lvalue<opaque> type, i.e.
// `T name;` / `std::array<T,N> name;`. We own this (rather than upstream
// MemRefToEmitC, which heap-allocs a pointer) so the buffer is a value that
// binds to the `std::array<T,N>&` boundaries and is subscriptable.
struct ConvertAllocLocal : public OpConversionPattern<mlir::memref::AllocOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::AllocOp op, OpAdaptor /*adaptor*/,
      ConversionPatternRewriter& rewriter) const override {
    Type converted = getTypeConverter()->convertType(op.getType());
    // payload buffer (lvalue<opaque>) or float C array (emitc.array): both are
    // valid emitc.variable result types -> a stack local. (We own this so float
    // allocs become stack arrays, not upstream's heap malloc/aligned_alloc.)
    if (!isa_and_present<emitc::LValueType>(converted) &&
        !isa_and_present<emitc::ArrayType>(converted))
      return failure();
    rewriter.replaceOpWithNewOp<emitc::VariableOp>(
        op, converted, emitc::OpaqueAttr::get(rewriter.getContext(), ""));
    return success();
  }
};

// memref.dealloc of a cheddar payload buffer (lvalue<opaque>, a move-only
// Ciphertext/Plaintext<word> that owns GPU memory) -> free the device buffer
// *now* by move-assigning an empty handle (`v = {};`), which runs the payload's
// destructor at last use. Without this, every intermediate stays live until the
// enclosing C++ function returns (scope-bound RAII), so peak GPU memory is the
// sum of ALL intermediates and large models OOM. The ownership-based buffer
// deallocation pass inserts these deallocs at last use; here we lower them.
struct EraseDealloc : public OpConversionPattern<mlir::memref::DeallocOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::DeallocOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Type memTy = adaptor.getMemref().getType();
    if (auto l = dyn_cast<emitc::LValueType>(memTy)) {
      if (auto opaque = dyn_cast<emitc::OpaqueType>(l.getValueType())) {
        // Move-only payload (Ciphertext/Plaintext<word>): free the GPU buffer
        // at last use by move-assigning a fresh default-constructed temporary,
        // e.g. `v = Ciphertext<word>();`. NB `v = {}` does NOT compile -- these
        // types are `explicit`-constructed (all-default-args ctor) and not
        // brace-assignable. The opaque type string IS the C++ type name.
        std::string reset = "{} = " + opaque.getValue().str() + "();";
        VerbatimOp::create(rewriter, op.getLoc(), reset,
                           ValueRange{adaptor.getMemref()});
      }
      // Non-payload lvalues (e.g. plain float scalars) are scope-bound stack
      // values with nothing to free -- just drop the dealloc.
      rewriter.eraseOp(op);
      return success();
    }
    // Plain float buffers lower to scope-bound stack arrays (emitc.array); the
    // ownership-based dealloc still inserts a memref.dealloc for them, which
    // has nothing to free -- drop it.
    if (isa<emitc::ArrayType>(memTy)) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

// memref.load on a std::array buffer (lvalue<opaque>) -> `base[i...]` via
// emitc.subscript. For a payload element the subscript (an lvalue) is fed to
// member_call operands directly; for a float element a memref.load yields a
// value, so add an emitc.load.
struct ConvertLoadArray : public OpConversionPattern<mlir::memref::LoadOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::LoadOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Type baseTy = adaptor.getMemref().getType();
    bool isPayloadBuf = isa<emitc::LValueType>(baseTy);
    if (!isPayloadBuf && !isa<emitc::ArrayType>(baseTy)) return failure();
    Type elt =
        getTypeConverter()->convertType(op.getMemRefType().getElementType());
    if (!elt) return failure();
    // Rank-0 payload memref: a rank-0 `memref<!cheddar.X>` converts to the
    // element lvalue itself (lvalue<opaque>), so a no-index load IS that lvalue
    // -- no subscript. Mirrors the indexed payload case which yields an lvalue
    // (used by preprocessing storage of a single plaintext slot). Non-payload
    // rank-0 loads fall through to the stock memref->emitc patterns.
    if (adaptor.getIndices().empty()) {
      if (isPayloadBuf && isa<emitc::OpaqueType>(elt)) {
        rewriter.replaceOp(op, adaptor.getMemref());
        return success();
      }
      return failure();
    }
    auto sub = emitc::SubscriptOp::create(
        rewriter, op.getLoc(), emitc::LValueType::get(elt), adaptor.getMemref(),
        adaptor.getIndices());
    if (isa<emitc::OpaqueType>(elt)) {
      rewriter.replaceOp(op, sub.getResult());  // payload: lvalue, no copy
      return success();
    }
    rewriter.replaceOpWithNewOp<emitc::LoadOp>(op, elt, sub.getResult());
    return success();
  }
};

// The dialect-conversion materialization for a payload memref.store's value
// operand turns an `lvalue<opaque<T>>` producer (an emitc.variable holding the
// payload) into the `opaque<T>` value form, via an unrealized_conversion_cast
// (the type converter maps payloads to the value type). For a move-only payload
// we want to `std::move` the LVALUE directly, so look through that cast.
// (reconcile-unrealized-casts can't fold it -- there's no inverse -- and an
// emitc.load would copy the move-only payload.)
static Value payloadMoveSource(Value v) {
  if (auto cast = v.getDefiningOp<mlir::UnrealizedConversionCastOp>();
      cast && cast.getInputs().size() == 1 &&
      isa<emitc::LValueType>(cast.getInputs()[0].getType()))
    return cast.getInputs()[0];
  return v;
}

// memref.store into a std::array buffer -> `base[i...] = ...`. A payload
// element is move-assigned (`arr[i] = std::move(src);`); a float element is a
// plain `emitc.assign`.
struct ConvertStoreArray : public OpConversionPattern<mlir::memref::StoreOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::StoreOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Type baseTy = adaptor.getMemref().getType();
    if (!isa<emitc::LValueType>(baseTy) && !isa<emitc::ArrayType>(baseTy))
      return failure();
    Type elt =
        getTypeConverter()->convertType(op.getMemRefType().getElementType());
    if (!elt) return failure();
    // Rank-0 payload memref: store directly into the element lvalue (no
    // subscript), mirroring the indexed payload move-assign below.
    if (adaptor.getIndices().empty()) {
      if (isa<emitc::LValueType>(baseTy) && isa<emitc::OpaqueType>(elt)) {
        VerbatimOp::create(rewriter, op.getLoc(), "{} = std::move({});",
                           ValueRange{adaptor.getMemref(),
                                      payloadMoveSource(adaptor.getValue())});
        rewriter.eraseOp(op);
        return success();
      }
      return failure();
    }
    auto sub = emitc::SubscriptOp::create(
        rewriter, op.getLoc(), emitc::LValueType::get(elt), adaptor.getMemref(),
        adaptor.getIndices());
    if (isa<emitc::OpaqueType>(elt)) {
      VerbatimOp::create(
          rewriter, op.getLoc(), "{} = std::move({});",
          ValueRange{sub.getResult(), payloadMoveSource(adaptor.getValue())});
    } else {
      emitc::AssignOp::create(rewriter, op.getLoc(), sub.getResult(),
                              adaptor.getValue());
    }
    rewriter.eraseOp(op);
    return success();
  }
};

// memref.copy of a cheddar buffer (both operands lvalue<opaque>). Move-only
// payload buffers are MOVED, not copied (a C++ copy is deleted); float buffers
// use std::array copy-assignment. (Payload copies appear after
// buffer-results-to-out-params when a function returns a buffer it didn't
// allocate in place; the source is dead afterward, so moving is correct.)
//   * payload scalar:  `dst = std::move(src);`
//   * payload array:   `std::move(std::begin(src), std::end(src),
//                                 std::begin(dst));`
//   * float (scalar/array): `dst = src;`  (std::array is copy-assignable)
struct ConvertCopy : public OpConversionPattern<mlir::memref::CopyOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::CopyOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value src = adaptor.getSource();
    Value tgt = adaptor.getTarget();
    auto opaqueOf = [](Value v) -> emitc::OpaqueType {
      auto l = dyn_cast<emitc::LValueType>(v.getType());
      return l ? dyn_cast<emitc::OpaqueType>(l.getValueType())
               : emitc::OpaqueType();
    };
    emitc::OpaqueType so = opaqueOf(src), to = opaqueOf(tgt);
    if (!so || !to) return failure();
    StringRef name = to.getValue();
    bool isArray = name.starts_with("std::array");
    if (opaqueNamesPayload(name)) {
      if (isArray)
        VerbatimOp::create(
            rewriter, op.getLoc(),
            "std::move(std::begin({}), std::end({}), std::begin({}));",
            ValueRange{src, src, tgt});
      else
        VerbatimOp::create(rewriter, op.getLoc(), "{} = std::move({});",
                           ValueRange{tgt, src});
    } else {
      VerbatimOp::create(rewriter, op.getLoc(), "{} = {};",
                         ValueRange{tgt, src});
    }
    rewriter.eraseOp(op);
    return success();
  }
};

// memref.subview slicing a std::array buffer (lvalue<opaque>) -> `base[o...]`,
// an lvalue subscript. The rank-reducing extract/insert slices used to pull a
// single ciphertext out of a `tensor<1x!cheddar.X>` packing container, or a row
// out of a `<1x1024xf32>` buffer, drop the leading (size-1) dims: emit a
// subscript indexing those dropped dims by their static offset, yielding the
// (converted) inner buffer / scalar.
struct ConvertSubViewSubscript
    : public OpConversionPattern<mlir::memref::SubViewOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::SubViewOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value base = adaptor.getSource();
    if (!isa<emitc::LValueType>(base.getType())) return failure();
    Type resultTy = getTypeConverter()->convertType(op.getType());
    if (!isa_and_present<emitc::LValueType>(resultTy)) return failure();
    int64_t srcRank = op.getSourceType().getRank();
    int64_t resRank = cast<MemRefType>(op.getType()).getRank();
    int64_t dropped = srcRank - resRank;
    if (dropped <= 0) return failure();
    // Subscript the dropped (leading) dims by their offsets. Offsets may be
    // static (a literal index) or dynamic (a loop induction variable, e.g. from
    // a `tensor.insert_slice` in an scf.for) -- emitc.subscript accepts dynamic
    // index operands, so thread the converted dynamic offset value through.
    auto mixedOffsets = op.getMixedOffsets();
    if (static_cast<int64_t>(mixedOffsets.size()) != srcRank) return failure();
    ValueRange dynOffsets = adaptor.getOffsets();
    auto sizeT = emitc::SizeTType::get(getContext());
    SmallVector<Value> idx;
    unsigned dynCursor = 0;
    for (int64_t i = 0; i < srcRank; ++i) {
      bool isDyn = isa<Value>(mixedOffsets[i]);
      if (i < dropped) {
        if (isDyn) {
          idx.push_back(dynOffsets[dynCursor]);
        } else {
          int64_t o =
              cast<IntegerAttr>(cast<Attribute>(mixedOffsets[i])).getInt();
          idx.push_back(emitc::LiteralOp::create(rewriter, op.getLoc(), sizeT,
                                                 std::to_string(o)));
        }
      }
      if (isDyn) ++dynCursor;
    }
    rewriter.replaceOpWithNewOp<emitc::SubscriptOp>(op, resultTy, base, idx);
    return success();
  }
};

// memref.cast of a payload buffer -> forward the converted source. The
// rank-reducing extract_slice that pulls a single ciphertext out of a
// `tensor<1x!cheddar.X>` packing container produces a `strided<[]>` rank-0
// memref; when that feeds an extern call (the __heir_debug read) whose decl arg
// is the plain unstrided memref, a `memref.cast` strided<[]> -> plain is
// inserted. Both sides are the same Ciphertext/Plaintext lvalue at the C++
// level, so the cast is a no-op: replace it with its (already converted)
// source.
struct ConvertPayloadCast : public OpConversionPattern<mlir::memref::CastOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::CastOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto resTy = dyn_cast<MemRefType>(op.getType());
    if (!resTy || !isa<cheddar::CiphertextType, cheddar::PlaintextType,
                       cheddar::ConstantType>(resTy.getElementType()))
      return failure();
    rewriter.replaceOp(op, adaptor.getSource());
    return success();
  }
};

// memref.subview producing a strided slice of a float C-array buffer ->
// `&base[o...]`, a raw pointer (the message slice fed to cheddar.encode).
struct ConvertSubViewToPointer
    : public OpConversionPattern<mlir::memref::SubViewOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::SubViewOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value base = adaptor.getSource();
    auto arrayTy = dyn_cast<emitc::ArrayType>(base.getType());
    if (!arrayTy || !isa<FloatType>(arrayTy.getElementType())) return failure();
    auto offsets = op.getStaticOffsets();
    if (static_cast<int64_t>(offsets.size()) != arrayTy.getShape().size())
      return failure();
    for (int64_t o : offsets)
      if (ShapedType::isDynamic(o)) return failure();
    auto sizeT = emitc::SizeTType::get(getContext());
    SmallVector<Value> idx;
    for (int64_t o : offsets)
      idx.push_back(emitc::LiteralOp::create(rewriter, op.getLoc(), sizeT,
                                             std::to_string(o)));
    auto lvalT = emitc::LValueType::get(arrayTy.getElementType());
    auto sub =
        emitc::SubscriptOp::create(rewriter, op.getLoc(), lvalT, base, idx);
    rewriter.replaceOpWithNewOp<emitc::AddressOfOp>(
        op, emitc::PointerType::get(arrayTy.getElementType()), sub.getResult());
    return success();
  }
};

// memref.expand_shape of a read-only whole float buffer -> a reshaped local
// array copy.
struct ConvertExpandShapeFloatCopy
    : public OpConversionPattern<mlir::memref::ExpandShapeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::ExpandShapeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto srcMemrefTy = cast<MemRefType>(op.getSrc().getType());
    auto resultMemrefTy = cast<MemRefType>(op.getResult().getType());
    if (!srcMemrefTy.hasStaticShape() || !resultMemrefTy.hasStaticShape())
      return failure();
    if (!isa<FloatType>(srcMemrefTy.getElementType()) ||
        srcMemrefTy.getElementType() != resultMemrefTy.getElementType())
      return failure();
    if (numElements(srcMemrefTy.getShape()) !=
        numElements(resultMemrefTy.getShape()))
      return failure();
    for (Operation* user : op.getResult().getUsers())
      if (isa<mlir::memref::StoreOp, mlir::memref::CopyOp,
              mlir::memref::SubViewOp>(user))
        return rewriter.notifyMatchFailure(
            op, "only read-only whole-buffer expand_shape is supported");
    auto srcArrayTy = dyn_cast<emitc::ArrayType>(adaptor.getSrc().getType());
    auto resultArrayTy = dyn_cast_or_null<emitc::ArrayType>(
        getTypeConverter()->convertType(resultMemrefTy));
    if (!srcArrayTy || !resultArrayTy) return failure();
    Value out = VariableOp::create(rewriter, op.getLoc(), resultArrayTy,
                                   OpaqueAttr::get(rewriter.getContext(), ""));
    Value outPtr = addressOfFirstElement(rewriter, op.getLoc(), out);
    Value srcPtr =
        addressOfFirstElement(rewriter, op.getLoc(), adaptor.getSrc());
    VerbatimOp::create(
        rewriter, op.getLoc(),
        "for (size_t _i = 0; _i < " +
            std::to_string(numElements(resultMemrefTy.getShape())) +
            "; ++_i) {}[_i] = {}[_i];",
        ValueRange{outPtr, srcPtr});
    rewriter.replaceOp(op, out);
    return success();
  }
};

// memref.copy between float buffers (C arrays / pointers) -> an element-wise
// loop (the move-only payload copy is handled by ConvertCopy).
struct ConvertMemRefCopyFloat
    : public OpConversionPattern<mlir::memref::CopyOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::CopyOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value src = adaptor.getSource();
    Value tgt = adaptor.getTarget();
    auto floatOperand = [](Value v) -> bool {
      if (auto p = dyn_cast<emitc::PointerType>(v.getType()))
        return isa<FloatType>(p.getPointee());
      if (auto a = dyn_cast<emitc::ArrayType>(v.getType()))
        return isa<FloatType>(a.getElementType());
      return false;
    };
    if (!floatOperand(src) || !floatOperand(tgt)) return failure();
    int64_t n = 1;
    if (auto sh = dyn_cast<ShapedType>(op.getSource().getType()))
      for (int64_t d : sh.getShape()) n *= d;
    // Index both buffers through flat `<elt>*` SSA pointers so a result
    // out-param target (later flattened to `float*` at the boundary) stays
    // valid; the printed C++ is unchanged for ordinary C-array copies.
    Value tgtPtr = addressOfFirstElement(rewriter, op.getLoc(), tgt);
    Value srcPtr = addressOfFirstElement(rewriter, op.getLoc(), src);
    std::string fmt = "for (size_t _i = 0; _i < " + std::to_string(n) +
                      "; ++_i) ({})[_i] = ({})[_i];";
    VerbatimOp::create(rewriter, op.getLoc(), fmt, ValueRange{tgtPtr, srcPtr});
    rewriter.eraseOp(op);
    return success();
  }
};

// Like upstream ConvertGlobal but tolerates an alignment attribute (bufferized
// constant float globals carry `alignment = 64`; emitc.global has no alignas).
// Float globals convert to emitc.array (a C array), which emitc.global accepts
// and emitc subscripts/get_globals natively.
struct ConvertGlobalDropAlign
    : public OpConversionPattern<mlir::memref::GlobalOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::GlobalOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    if (!op.getType().hasStaticShape()) return failure();
    Type resultTy = getTypeConverter()->convertType(op.getType());
    if (!isa_and_present<emitc::ArrayType>(resultTy)) return failure();
    auto vis = SymbolTable::getSymbolVisibility(op);
    if (vis != SymbolTable::Visibility::Public &&
        vis != SymbolTable::Visibility::Private)
      return failure();
    bool staticSpecifier = vis == SymbolTable::Visibility::Private;
    Attribute initialValue = adaptor.getInitialValueAttr();
    if (isa_and_present<UnitAttr>(initialValue)) initialValue = {};
    // Emit the global non-const even when the source memref is `constant`.
    // A read-only float buffer arg (e.g. `_assign_layout`'s input) prints as a
    // plain `float v[1][512]` param -- func.func args carry no const qualifier
    // in this emitter -- so a `static const float[...]` global cannot bind to
    // it (`no matching function`). The globals are machine-generated and never
    // mutated, so dropping `const` is safe and keeps them callable everywhere.
    rewriter.replaceOpWithNewOp<emitc::GlobalOp>(
        op, adaptor.getSymName(), resultTy, initialValue,
        /*externSpecifier=*/!staticSpecifier, staticSpecifier,
        /*constSpecifier=*/false);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertToEmitC dialect interface
//===----------------------------------------------------------------------===//

// The func dialect *promises* ConvertToEmitCPatternInterface, and
// --convert-to-emitc dyn_casts every loaded dialect to it (a hard error if a
// promise is unimplemented). We must NOT use the stock FuncToEmitC (it forms
// `emitc.func`, which cannot carry the move-only payload `lvalue` args); the
// cheddar interface instead keeps `func.func` via a structural type conversion.
// So we satisfy func's promise with this no-op implementation.
struct NoOpToEmitCInterface : public ConvertToEmitCPatternInterface {
  NoOpToEmitCInterface(Dialect* dialect)
      : ConvertToEmitCPatternInterface(dialect) {}
  void populateConvertToEmitCConversionPatterns(
      ConversionTarget&, TypeConverter&, RewritePatternSet&,
      std::optional<bool>) const final {}
};

// Populate target legality, type conversions, and patterns for lowering cheddar
// (plus the func-boundary structural conversion that keeps `func.func`) to
// EmitC. Driven by `--convert-to-emitc` (which also pulls in arith/scf/memref
// via their own interfaces).
struct CheddarToEmitCDialectInterface : public ConvertToEmitCPatternInterface {
  CheddarToEmitCDialectInterface(Dialect* dialect)
      : ConvertToEmitCPatternInterface(dialect) {}

  void populateConvertToEmitCConversionPatterns(
      ConversionTarget& target, TypeConverter& typeConverter,
      RewritePatternSet& patterns,
      std::optional<bool> /*lowerToCpp*/) const final {
    MLIRContext* ctx = patterns.getContext();
    addCheddarEmitCTypeConversions(typeConverter, ctx);

    // Keep func.func/return/call (structural type conversion only). The SCF
    // interface sets markUnknownOpDynamicallyLegal(true), so set the legality
    // we depend on explicitly rather than relying on defaults.
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    // Gate on signature legality only: requiring the body to also be legal
    // creates a circular dependency (the driver re-checks the func before
    // descending into the body), leaving the func "updated in place but still
    // illegal". Body ops are converted independently by their own patterns.
    target.addDynamicallyLegalOp<func::FuncOp>(
        [&typeConverter](func::FuncOp op) {
          return typeConverter.isSignatureLegal(op.getFunctionType());
        });
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&typeConverter](func::ReturnOp op) {
          return typeConverter.isLegal(op);
        });
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::CallOp>(
        [&typeConverter](func::CallOp op) {
          // A __heir_debug_* call is rewritten to an emitc.call_opaque
          // "__heir_debug" by ConvertDebugCall; never treat it as legal.
          return !isDebugPort(op.getCallee()) && typeConverter.isLegal(op);
        });
    patterns.add<ConvertDebugCall>(typeConverter, ctx, /*benefit=*/2);

    target.addIllegalDialect<cheddar::CheddarDialect>();
    target.addIllegalDialect<arith::ArithDialect>();
    target.addDynamicallyLegalDialect<mlir::memref::MemRefDialect>(
        [&typeConverter](Operation* op) { return typeConverter.isLegal(op); });
    // memref.global has no typed operand/result, so the dialect-level isLegal
    // check above always considers it legal and never converts it -- leaving
    // the converted emitc.get_global referencing a missing emitc.global. Force
    // it illegal so ConvertGlobalDropAlign lowers it.
    target.addIllegalOp<mlir::memref::GlobalOp>();

    // The cheddar interface owns memref->emitc lowering (the memref dialect's
    // own interface is a no-op; see registerCheddarConvertToEmitCInterface).
    // Pull in the stock MemRefToEmitC patterns for the plain float ops
    // (alloc/load/store/global) at default benefit; the custom payload/float
    // patterns below at benefit 2 win where they apply. (This helper does not
    // touch the type converter -- the memref type conversions live in
    // addCheddarEmitCTypeConversions.)
    mlir::populateMemRefToEmitCConversionPatterns(patterns, typeConverter);

    // Payload memref ops + float-buffer ops (benefit 2 to win over the stock
    // MemRefToEmitC patterns added just above).
    patterns.add<ConvertAllocLocal, EraseDealloc, ConvertLoadArray,
                 ConvertStoreArray, ConvertCopy, ConvertMemRefCopyFloat,
                 ConvertSubViewSubscript, ConvertSubViewToPointer,
                 ConvertPayloadCast, ConvertExpandShapeFloatCopy,
                 ConvertGlobalDropAlign>(typeConverter, ctx, /*benefit=*/2);

    patterns.add<ConvertMakeParameter, ConvertPrepareRotKey,
                 ConvertCreateBootContext, ConvertPrepareBootstrap,
                 ConvertEncode, ConvertEncodeConstant, ConvertDecode,
                 ConvertHRot, ConvertHRotAdd, ConvertHConj, ConvertHConjAdd,
                 ConvertLinearTransform, ConvertPrepareLinearTransform,
                 ConvertApplyPreparedLinearTransform, ConvertEvalPoly>(
        typeConverter, ctx);
    patterns.add<ConvertSetupAssign<cheddar::CreateContextOp>>(
        typeConverter, ctx, "Context<word>::Create");
    patterns.add<ConvertSetupAssign<cheddar::CreateUserInterfaceOp>>(
        typeConverter, ctx, "std::make_unique<UserInterface<word>>");

    auto addDps = [&](StringRef name, auto opTag,
                      std::function<std::string(decltype(opTag))> extra =
                          nullptr) {
      using Op = decltype(opTag);
      patterns.add<OutParamDpsPattern<Op>>(typeConverter, ctx, name, extra);
    };
    addDps("Add", cheddar::AddOp{});
    addDps("Sub", cheddar::SubOp{});
    addDps("Mult", cheddar::MultOp{});
    addDps("Add", cheddar::AddPlainOp{});
    addDps("Sub", cheddar::SubPlainOp{});
    addDps("Mult", cheddar::MultPlainOp{});
    addDps("Add", cheddar::AddConstOp{});
    addDps("Mult", cheddar::MultConstOp{});
    addDps("Neg", cheddar::NegOp{});
    addDps("Rescale", cheddar::RescaleOp{});
    addDps("Relinearize", cheddar::RelinearizeOp{});
    addDps("RelinearizeRescale", cheddar::RelinearizeRescaleOp{});
    addDps("Encrypt", cheddar::EncryptOp{});
    addDps("Decrypt", cheddar::DecryptOp{});
    addDps("MadUnsafe", cheddar::MadUnsafeOp{});
    addDps("Boot", cheddar::BootOp{});
    addDps("LevelDown", cheddar::LevelDownOp{}, [](cheddar::LevelDownOp op) {
      return intLit(op.getTargetLevelAttr());
    });
    addDps("HMult", cheddar::HMultOp{}, [](cheddar::HMultOp op) {
      return op.getRescale() ? std::string("true") : std::string("false");
    });
  }
};

//===----------------------------------------------------------------------===//
// cheddar-emitc-boundary pass
//===----------------------------------------------------------------------===//

// Build the C++ reference type for a converted buffer arg. Every cheddar buffer
// converts to `lvalue<opaque T>` (T a scalar payload or a `std::array<...>`):
//   lvalue<opaque T> -> `T&` (written) / `const T&` (read-only).
// A func/emitc.func cannot carry an lvalue arg, so this is applied to the
// boundary by the cheddar-emitc-boundary pass.
Type referenceArgType(MLIRContext* ctx, Type converted, bool written) {
  // Payload buffer args: lvalue<opaque> -> `T&` / `const T&` (T is a scalar
  // payload or a std::array of payloads).
  if (auto l = dyn_cast<emitc::LValueType>(converted)) {
    auto o = dyn_cast<emitc::OpaqueType>(l.getValueType());
    if (!o) return {};
    std::string base = o.getValue().str();
    return OpaqueType::get(ctx,
                           written ? (base + "&") : ("const " + base + "&"));
  }
  // Non-copyable handle args passed by value would force a move/copy at the
  // call boundary (Encoder is a reference-holding view; EvkMap and
  // EvaluationKey are move-only) and mismatch the C++ harness's `const T&`
  // declarations. They only ever appear as read-only inputs, so tighten to
  // `const T&`.
  if (auto o = dyn_cast<emitc::OpaqueType>(converted)) {
    StringRef n = o.getValue();
    if (n == "Encoder<word>" || n == "EvkMap<word>" ||
        n == "EvaluationKey<word>")
      return OpaqueType::get(ctx, ("const " + n + "&").str());
  }
  return {};
}

// A float-element *result* out-param (carries `bufferize.result`) must match
// the C++ harness's flat `float*` convention -- the pre-DPS emitter lifted
// float-array results to `float*`. buffer-results-to-out-params instead hands
// us a multi-dim C array (`float v[1][10]`), whose param type decays to
// `float(*)[10]` != `float*` and fails to link. Retype such args to `<elt>*`
// and rewrite the body's `&arg[0]..[0]` (addressOfFirstElement -- the only way
// the decode copy touches the buffer) to `arg`. Float *inputs* (images,
// weights; no `bufferize.result`) keep their multi-dim shape. Returns the
// pointer type, or {} if the arg isn't a flat-able float-array result.
Type flattenFloatResultArg(BlockArgument arg) {
  auto arr = dyn_cast<emitc::ArrayType>(arg.getType());
  if (!arr || !isa<FloatType>(arr.getElementType())) return {};
  // Every use must be `address_of(subscript(arg, 0..0))`, so flattening to a
  // pointer is sound (a residual multi-index subscript on a pointer would be
  // invalid C++); otherwise bail and leave the arg as a C array.
  SmallVector<std::pair<emitc::AddressOfOp, emitc::SubscriptOp>> toRewrite;
  for (OpOperand& use : arg.getUses()) {
    auto sub = dyn_cast<emitc::SubscriptOp>(use.getOwner());
    if (!sub || sub.getValue() != arg || !sub.getResult().hasOneUse())
      return {};
    auto addr =
        dyn_cast<emitc::AddressOfOp>(*sub.getResult().getUsers().begin());
    if (!addr) return {};
    toRewrite.push_back({addr, sub});
  }
  auto ptrTy = emitc::PointerType::get(arr.getElementType());
  arg.setType(ptrTy);
  for (auto& [addr, sub] : toRewrite) {
    addr.getResult().replaceAllUsesWith(arg);
    addr.erase();
    sub.erase();
  }
  return ptrTy;
}

// A value is "written" if used as the destination (out) of an emitted call:
// member_call_opaque places the dest at operand 1 (after the receiver), a
// call_opaque shim (RunLinearTransform/RunEvalPoly) at operand 0, or an
// assignment verbatim (`{} = ...`) at operand 0. A move-only payload that is
// the *source* of a `std::move(...)` verbatim (emitted by ConvertCopy when a
// function returns a buffer it didn't allocate, e.g. a returned-unchanged arg)
// must likewise be a non-const lvalue -- `std::move` cannot bind a `const T&`
// -- so any operand of a move verbatim counts as written.
bool valueWrittenAsDest(Value v) {
  for (OpOperand& use : v.getUses()) {
    Operation* owner = use.getOwner();
    unsigned idx = use.getOperandNumber();
    if (isa<MemberCallOpaqueOp>(owner) && idx == 1) return true;
    if (isa<CallOpaqueOp>(owner) && idx == 0) return true;
    if (auto vb = dyn_cast<VerbatimOp>(owner)) {
      StringRef s = vb.getValue();
      if ((idx == 0 && s.contains("=")) || s.contains("std::move")) return true;
    }
  }
  return false;
}

// An out-param arg carries the `bufferize.result` attr; an in-place arg is
// detected from its body uses (directly or via a subscript).
bool isPayloadArgWritten(func::FuncOp fn, unsigned i) {
  if (fn.getArgAttr(i, "bufferize.result")) return true;
  BlockArgument arg = fn.getBody().front().getArgument(i);
  if (valueWrittenAsDest(arg)) return true;
  for (Operation* u : arg.getUsers())
    if (auto sub = dyn_cast<emitc::SubscriptOp>(u))
      if (sub.getValue() == arg && valueWrittenAsDest(sub.getResult()))
        return true;
  return false;
}

// Evaluation-key lookup shims, emitted once at module scope (unconditionally:
// any program that rotates, conjugates or multiplies needs them).
//
// cyclops from CYC-173 ("Residual/boot parameter split + ring & secret
// identity") indexes every evaluation key by a SecretId and takes an explicit
// level on rotation lookups; scale-snu CHEDDAR keeps the older secret-less
// signatures (GetRotationKey(int), GetConjugationKey(),
// GetMultiplicationKey()). Tag-dispatch on whichever expression compiles, same
// int/long idiom as PrepareLintransRotKey below, so one emitted TU serves both
// forks.
//
// The secret is always taken from the operand ciphertext's own tag, which is
// how cyclops itself does it (src/extension/jkls18/Matmul.cpp,
// src/extension/nn/Pooling.cpp) and what core/Type.h prescribes: containers
// carry a SecretId and key resolution matches the ciphertext's tag.
//
// level is passed as -1, cyclops' documented "unconstrained" lookup: EvkMap
// asserts level == -1 || 0 <= level <= max_level_ and then selects the
// compatible key with the most Q primes, i.e. the chain-max key CHEDDAR uses at
// any level. This deliberately avoids threading per-op levels through the
// cheddar dialect, whose ops carry no level attribute. NB: it also means
// lookups do not exercise cyclops' level-specific key selection.
constexpr llvm::StringLiteral kKeyLookupShim = R"cpp(
#include <type_traits>
#include <utility>
  // A ciphertext's SecretId, whether the emitted value is an object or pointer.
  template <typename T>
  static auto HeirSecretOf(const T& ct, int) -> decltype(ct.GetSecretId()) {
    return ct.GetSecretId();
  }
  template <typename T>
  static auto HeirSecretOf(T* ct, long) -> decltype(ct->GetSecretId()) {
    return ct->GetSecretId();
  }
  // Rotation key: (d, level=-1, secret) on cyclops, (d) on scale-snu CHEDDAR.
  template <typename UIP, typename CtT>
  static auto HeirRotationKeyImpl(UIP& ui, int d, const CtT& in, int)
      -> decltype(ui->GetRotationKey(d, -1, HeirSecretOf(in, 0))) {
    return ui->GetRotationKey(d, -1, HeirSecretOf(in, 0));
  }
  template <typename UIP, typename CtT>
  static auto HeirRotationKeyImpl(UIP& ui, int d, const CtT&,
                                  long) -> decltype(ui->GetRotationKey(d)) {
    return ui->GetRotationKey(d);
  }
  template <typename UIP, typename CtT>
  static decltype(auto) HeirRotationKey(UIP& ui, int d, const CtT& in) {
    return HeirRotationKeyImpl(ui, d, in, 0);
  }
  // Conjugation key. Not just an arity change: cyclops has no
  // UserInterface::GetConjugationKey at all (only a private
  // PrepareConjugationKey) and the lookup lives on the EvkMap, reached through
  // the public GetEvkMap() -- which is how cyclops' own BootContext.cpp does it
  // (evk_map.GetConjugationKey(ct.GetSecretId())). scale-snu CHEDDAR keeps the
  // zero-arg accessor directly on the UserInterface.
  template <typename UIP, typename CtT>
  static auto HeirConjugationKeyImpl(UIP& ui, const CtT& in, int)
      -> decltype(ui->GetEvkMap().GetConjugationKey(HeirSecretOf(in, 0))) {
    return ui->GetEvkMap().GetConjugationKey(HeirSecretOf(in, 0));
  }
  template <typename UIP, typename CtT>
  static auto HeirConjugationKeyImpl(UIP& ui, const CtT&, long)
      -> decltype(ui->GetConjugationKey()) {
    return ui->GetConjugationKey();
  }
  template <typename UIP, typename CtT>
  static decltype(auto) HeirConjugationKey(UIP& ui, const CtT& in) {
    return HeirConjugationKeyImpl(ui, in, 0);
  }
  // Multiplication key off an EvkMap-like holder: (secret) on cyclops, () on
  // scale-snu CHEDDAR. Taken from the operand being multiplied.
  template <typename EM, typename CtT>
  static auto HeirMultiplicationKeyImpl(EM& em, const CtT& in, int)
      -> decltype(em.GetMultiplicationKey(HeirSecretOf(in, 0))) {
    return em.GetMultiplicationKey(HeirSecretOf(in, 0));
  }
  template <typename EM, typename CtT>
  static auto HeirMultiplicationKeyImpl(EM& em, const CtT&, long)
      -> decltype(em.GetMultiplicationKey()) {
    return em.GetMultiplicationKey();
  }
  template <typename EM, typename CtT>
  static decltype(auto) HeirMultiplicationKey(EM& em, const CtT& in) {
    return HeirMultiplicationKeyImpl(em, in, 0);
  }
  // Keygen-side rotation key. Unlike the lookups above there is no ciphertext
  // to take a tag from, so cyclops' SecretId comes from the Parameter behind
  // the Context (Context::param_ is public; cyclops' own extension code uses
  // context_->param_ the same way). The residual secret is the one evaluation
  // keys are built under -- the residual/boot split is what introduced these
  // signatures. scale-snu CHEDDAR keeps (rot, max_level).
  template <typename UIP, typename CtxP>
  static auto HeirPrepareRotKeyImpl(UIP& ui, int d, int max_level,
                                    const CtxP& ctx, int)
      -> decltype(ui->PrepareRotationKey(d, ctx->param_.ResidualSecretId(),
                                         max_level),
                  void()) {
    ui->PrepareRotationKey(d, ctx->param_.ResidualSecretId(), max_level);
  }
  template <typename UIP, typename CtxP>
  static void HeirPrepareRotKeyImpl(UIP& ui, int d, int max_level, const CtxP&,
                                    long) {
    ui->PrepareRotationKey(d, max_level);
  }
  template <typename UIP, typename CtxP>
  static void HeirPrepareRotKey(UIP& ui, int d, int max_level,
                                const CtxP& ctx) {
    HeirPrepareRotKeyImpl(ui, d, max_level, ctx, 0);
  }
  // Bootstrap rotation keys from an EvkRequest. Boot keys are built under the
  // boot secret, not the residual one, so this takes BootSecretId() (matching
  // cyclops' own test/bench usage) rather than the residual secret above.
  template <typename UIP, typename Req, typename CtxP>
  static auto HeirPrepareBootRotKeysImpl(UIP& ui, Req& req, const CtxP& ctx,
                                         int)
      -> decltype(ui->PrepareRotationKey(req, ctx->BootSecretId()), void()) {
    ui->PrepareRotationKey(req, ctx->BootSecretId());
  }
  template <typename UIP, typename Req, typename CtxP>
  static void HeirPrepareBootRotKeysImpl(UIP& ui, Req& req, const CtxP&, long) {
    ui->PrepareRotationKey(req);
  }
  template <typename UIP, typename Req, typename CtxP>
  static void HeirPrepareBootRotKeys(UIP& ui, Req& req, const CtxP& ctx) {
    HeirPrepareBootRotKeysImpl(ui, req, ctx, 0);
  }
)cpp";

// The cheddar.linear_transform lowering calls this shim, emitted once at
// module scope so generated kernels are self-contained (no consumer prelude
// copy). Backend headers (extension/linalg/{LinearTransform,StripedMatrix}.h
// or a test stub) are the consumer's to include — their layout is
// backend-specific — hence only std includes and fully qualified names here.
constexpr llvm::StringLiteral kRunLinearTransformShim = R"cpp(
  // cheddar.linear_transform shim (emitted by cheddar-emitc-boundary): pack the
  // diagonal-packed weights into a StripedMatrix, construct the transform at
  // the op's level/scale, evaluate with CHEDDAR's hoisted BSGS kernel. diagT:
  // f64 (orion frontend) or f32 (torch-linalg path). Construction encodes every
  // diagonal (the dominant cost); the weights are process-lifetime constants,
  // so the transform is memoized on the diagonal array's address.
#include <algorithm>
#include <complex>
#include <initializer_list>
#include <map>
#include <memory>
#include <set>
#include <type_traits>
#include <vector>
  // MinKS (minimal key-switching) eligibility. scale-snu CHEDDAR's Evaluate /
  // AddRequiredRotations assert (abort) with min_ks=true unless the transform's
  // nonzero baby AND giant rotations each form a COMPLETE arithmetic
  // progression. IsUsingBSGS() alone (bs>1 && gs>1) is too permissive: a
  // gap-structured transform (e.g. a dilated conv1d's expanded Toeplitz)
  // satisfies it yet fails the progression check -> std::exit(1). Compute the
  // check on the diagonal map once at construction, memoize per transform, and
  // use it identically for keygen and eval so a needed key is never missing.
  static int LinearTransformGCD(int a, int b) {
    while (b != 0) {
      int r = a % b;
      a = b;
      b = r;
    }
    return a < 0 ? -a : a;
  }
  static std::map<const void*, bool>& LinearTransformMinKSMap() {
    static auto* modes = new std::map<const void*, bool>();
    return *modes;
  }
  template <typename Matrix>
  static bool CanUseLinearTransformMinKS(const Matrix& matrix, int W, int bs,
                                         int gs) {
    if (bs <= 1 || gs <= 1) return false;
    int stride = 0;
    for (const auto& item : matrix) {
      int rot = item.first % W;
      if (rot < 0) rot += W;
      stride = LinearTransformGCD(stride, rot);
    }
    if (stride == 0) return false;
    int gs_stride = stride * bs;
    std::set<int> bs_indices;
    std::set<int> gs_indices;
    for (const auto& item : matrix) {
      int rot = item.first % W;
      if (rot < 0) rot += W;
      int bs_rot = rot % gs_stride;
      bs_indices.insert(bs_rot);
      gs_indices.insert(rot - bs_rot);
    }
    auto is_complete_stride = [](const std::set<int>& indices) {
      int count = 0;
      int gcd = 0;
      int maximum = 0;
      for (int index : indices) {
        if (index == 0) continue;
        gcd = LinearTransformGCD(gcd, index);
        maximum = std::max(maximum, index);
        ++count;
      }
      return count > 0 && count * gcd == maximum;
    };
    return is_complete_stride(bs_indices) && is_complete_stride(gs_indices);
  }
  static bool LinearTransformUsesMinKS(const void* transform) {
    auto& modes = LinearTransformMinKSMap();
    auto it = modes.find(transform);
    return it != modes.end() && it->second;
  }
  // Evaluate with min_ks=true when the fork supports it: scale-snu CHEDDAR's
  // hoisted BSGS silently corrupts transforms at deep levels (beta == 1) when
  // key-switching with chain-max keys, and its EvkMap cannot hold per-level
  // keys; min_ks selects its non-hoisted per-rotation key-switch, which (like
  // a plain hrot) is correct at any level with chain-max keys. Cyclops'
  // Evaluate has no such flag and handles levels correctly on its own, so the
  // 4-argument overload is selected there.
  template <typename LT, typename CP, typename Ct, typename EM>
  static auto RunLinearTransformEval(const LT& lt, CP cp, Ct& out, const Ct& in,
                                     const EM& em, int)
      -> decltype(lt.Evaluate(cp, out, in, em, true), void()) {
    // min_ks only when the rotations form complete arithmetic progressions
    // (see CanUseLinearTransformMinKS); IsUsingBSGS() alone abort()s scale-snu
    // on gap-structured transforms.
    lt.Evaluate(cp, out, in, em, /*min_ks=*/LinearTransformUsesMinKS(&lt));
  }
  template <typename LT, typename CP, typename Ct, typename EM>
  static void RunLinearTransformEval(const LT& lt, CP cp, Ct& out, const Ct& in,
                                     const EM& em, long) {
    lt.Evaluate(cp, out, in, em);
  }
  // Rotation-key prep for linear-transform rotations, dispatched on
  // PrepareRotationKey's arity: cyclops has a (rot, level, force, ...)
  // overload and builds level-specific keys (its best-fit lookup wants the
  // exact per-level key-switch config; keys live under distinct indices);
  // scale-snu cheddar only has (rot, level), holds ONE key per rotation
  // index (a re-prep at another level overwrites it and crashes), and its
  // min_ks evaluation is correct with chain-max keys at any level, so
  // everything is prepared at chain max there and duplicates dedupe.
  // Since CYC-173 cyclops' arity is (rot, secret, max_level, force, ...), so
  // the preferred overload is gated on that signature and takes the residual
  // secret from the Context's Parameter (see HeirPrepareRotKey above).
  template <typename UIP, typename CtxP>
  static auto PrepareLintransRotKeyImpl(UIP& ui, int d, int level,
                                        int chain_max, const CtxP& ctx, int)
      -> decltype(ui->PrepareRotationKey(d, ctx->param_.ResidualSecretId(),
                                         level, false),
                  void()) {
    ui->PrepareRotationKey(d, ctx->param_.ResidualSecretId(), level);
  }
  template <typename UIP, typename CtxP>
  static void PrepareLintransRotKeyImpl(UIP& ui, int d, int level,
                                        int chain_max, const CtxP&, long) {
    ui->PrepareRotationKey(d, chain_max);
  }
  template <typename UIP, typename CtxP>
  static void PrepareLintransRotKey(UIP& ui, int d, int level, int chain_max,
                                    const CtxP& ctx) {
    PrepareLintransRotKeyImpl(ui, d, level, chain_max, ctx, 0);
  }
  // Construct a LinearTransform, passing a compact per-prime plaintext period
  // when the fork's constructor supports it. An arbitrary W-slot CKKS message
  // has only a 2*W-word period in cyclops' bit-reversed NTT plaintext layout,
  // so keeping a single period (log_pt_size_per_prime = floor(log2 W) + 1)
  // instead of the full ring-degree plaintext sharply cuts a prepared
  // transform's device residency (the LogN-16 driver of GPU OOM). scale-snu
  // CHEDDAR has no such constructor argument; SFINAE selects its base
  // constructor there (mirrors the RunLinearTransformEval dispatch above).
  static int LinearTransformLogPtSizePerPrime(int W) {
    int lps = 1;
    for (int n = W; n > 1; n >>= 1) ++lps;
    return lps;
  }
  template <typename LT, typename CP, typename M>
  static auto MakeLinearTransform(CP cp, const M& m, int level, double scale,
                                  int bs, int gs, int logPtSizePerPrime, int)
      -> std::enable_if_t<std::is_constructible_v<LT, CP, M, int, double, int,
                                                  int, int, int, int>,
                          std::shared_ptr<LT>> {
    // NB: constrain on is_constructible_v, NOT decltype(make_shared<LT>(...)):
    // make_shared is variadic, so its return type is well-formed for ANY args
    // and the ctor-viability check happens in its (non-immediate) body -> that
    // decltype never SFINAEs, so it would always pick this 9-arg overload and
    // hard-error on scale-snu cheddar's 8-arg ctor. is_constructible_v tests
    // the ctor in the immediate context, so it correctly falls back below.
    return std::make_shared<LT>(cp, m, level, scale, bs, gs,
                                /*pre_rotation=*/0, /*additional_pt_rot=*/0,
                                logPtSizePerPrime);
  }
  template <typename LT, typename CP, typename M>
  static std::shared_ptr<LT> MakeLinearTransform(CP cp, const M& m, int level,
                                                 double scale, int bs, int gs,
                                                 int /*logPtSizePerPrime*/,
                                                 long) {
    return std::make_shared<LT>(cp, m, level, scale, bs, gs);
  }
  // Deferred linear-transform keygen: ask a prepared transform for exactly the
  // rotations it needs (post zero-diagonal pruning) so the harness can generate
  // only those keys, instead of __configure emitting the full conservative BSGS
  // set. scale-snu cheddar's AddRequiredRotations takes (req, min_ks) and must
  // use the SAME min_ks decision as evaluation (RunLinearTransformEval passes
  // lt.IsUsingBSGS()), or keygen and eval disagree and a needed key is missing;
  // cyclops exposes only AddRequiredRotations(req) and needs no such flag.
  // Prefer the 2-arg form (the (int) overload) so cheddar matches its eval;
  // cyclops (no 2-arg / no IsUsingBSGS) falls to the (long) 1-arg overload.
  template <typename T, typename Req>
  static auto AddLintransRequiredRotations(const std::shared_ptr<T>& lt,
                                           Req& req, int)
      -> decltype(lt->AddRequiredRotations(req, lt->IsUsingBSGS()), void()) {
    // Same min_ks decision as eval (RunLinearTransformEval) so keygen requests
    // exactly the keys eval uses; IsUsingBSGS() would over-approximate and
    // abort scale-snu on gap-structured transforms.
    lt->AddRequiredRotations(req, LinearTransformUsesMinKS(lt.get()));
  }
  template <typename T, typename Req>
  static void AddLintransRequiredRotations(const std::shared_ptr<T>& lt,
                                           Req& req, long) {
    lt->AddRequiredRotations(req);
  }
  template <int W, typename wordT, typename diagT>
  static void PrepareLinearTransform(
      std::shared_ptr<cheddar::LinearTransform<wordT>>& out,
      cheddar::Context<wordT>* ctx, diagT diag[][W],
      std::initializer_list<int> idx, int level, int bs, int gs) {
    cheddar::StripedMatrix m(W, W);
    int d = 0;
    for (int k : idx) {
      m[k] = std::vector<std::complex<double>>(diag[d], diag[d] + W);
      ++d;
    }
    cheddar::ConstContextPtr<wordT> cp(cheddar::ConstContextPtr<wordT>(), ctx);
    out = MakeLinearTransform<cheddar::LinearTransform<wordT>>(
        cp, m, level, ctx->param_.GetScale(level), bs, gs,
        LinearTransformLogPtSizePerPrime(W), 0);
    // Memoize MinKS eligibility keyed on the transform pointer; keygen and eval
    // both look it up (no-op on cyclops, which ignores the min_ks flag).
    LinearTransformMinKSMap()[out.get()] =
        CanUseLinearTransformMinKS(m, W, bs, gs);
  }
  template <typename wordT>
  static void RunPreparedLinearTransform(
      cheddar::Ciphertext<wordT>& out, cheddar::Context<wordT>* ctx,
      const cheddar::Ciphertext<wordT>& in,
      const cheddar::EvkMap<wordT>& evk_map,
      const std::shared_ptr<cheddar::LinearTransform<wordT>>& transform) {
    cheddar::ConstContextPtr<wordT> cp(cheddar::ConstContextPtr<wordT>(), ctx);
    RunLinearTransformEval(*transform, cp, out, in, evk_map, 0);
    // CHEDDAR forks differ on whether Evaluate rescales internally (cyclops
    // does via ModDownAndRescale; scale-snu cheddar leaves scale^2).
    if (out.GetNP().num_main_ == in.GetNP().num_main_) {
      cheddar::Ciphertext<wordT> rescaled;
      ctx->Rescale(rescaled, out);
      out = std::move(rescaled);
    }
  }
  template <int W, typename wordT, typename diagT>
  static void RunLinearTransform(cheddar::Ciphertext<wordT>& out,
                                 cheddar::Context<wordT>* ctx,
                                 const cheddar::Ciphertext<wordT>& in,
                                 const cheddar::EvkMap<wordT>& evk_map,
                                 diagT diag[][W],
                                 std::initializer_list<int> idx, int level,
                                 int bs, int gs) {
    // Heap-allocated and intentionally never destroyed: cached transforms
    // hold GPU resources, and static teardown runs after the CUDA context is
    // gone (destructing then SIGSEGVs on process exit).
    static auto* cache =
        new std::map<const void*, cheddar::LinearTransform<wordT>>();
    auto it = cache->find(diag);
    if (it == cache->end()) {
      cheddar::ConstContextPtr<wordT> cp(cheddar::ConstContextPtr<wordT>(), ctx);
      cheddar::StripedMatrix m(W, W);
      int d = 0;
      for (int k : idx) {
        m[k] = std::vector<std::complex<double>>(diag[d], diag[d] + W);
        ++d;
      }
      it = cache
               ->emplace(std::piecewise_construct, std::forward_as_tuple(diag),
                         std::forward_as_tuple(
                             cp, m, level, ctx->param_.GetScale(level), bs, gs))
               .first;
    }
    auto borrowed = std::shared_ptr<cheddar::LinearTransform<wordT>>(
        &it->second, [](cheddar::LinearTransform<wordT>*) {});
    RunPreparedLinearTransform(out, ctx, in, evk_map, borrowed);
  }
)cpp";

struct CheddarToEmitCPass
    : public impl::CheddarToEmitCBase<CheddarToEmitCPass> {
  using CheddarToEmitCBase::CheddarToEmitCBase;

  void runOnOperation() override {
    auto* ctx = &getContext();
    if (diagnoseUnsupportedGetters(getOperation())) {
      signalPassFailure();
      return;
    }

    // Generated kernels carry the RunLinearTransform shim themselves; emit it
    // once at module scope when any lowering produced a call to it. Idempotent
    // so pipelines that run this pass more than once (e.g. a separate emitc
    // lowering invocation after scheme-to-cheddar's) don't redefine it.
    auto module = cast<ModuleOp>(getOperation());
    bool usesLinearTransform = false;
    module->walk([&](emitc::CallOpaqueOp call) {
      if (call.getCallee().contains("LinearTransform"))
        usesLinearTransform = true;
    });
    // The __configure key prep also calls into the shim (the
    // PrepareLintransRotKey fork dispatch), emitted as a verbatim call.
    module->walk([&](emitc::VerbatimOp verbatim) {
      if (verbatim.getValue().starts_with("PrepareLintransRotKey"))
        usesLinearTransform = true;
    });
    bool shimAlreadyEmitted = false;
    bool keyShimAlreadyEmitted = false;
    for (auto verbatim : module.getBody()->getOps<emitc::VerbatimOp>()) {
      if (verbatim.getValue().contains("RunLinearTransform"))
        shimAlreadyEmitted = true;
      if (verbatim.getValue().contains("HeirRotationKeyImpl"))
        keyShimAlreadyEmitted = true;
    }
    if (usesLinearTransform && !shimAlreadyEmitted) {
      OpBuilder builder(ctx);
      builder.setInsertionPointToStart(module.getBody());
      emitc::VerbatimOp::create(builder, module.getLoc(),
                                kRunLinearTransformShim);
    }
    // The key-lookup shims are emitted unconditionally: unlike the linear
    // transform, rotation/conjugation/multiplication lookups appear in almost
    // every program, and the shim is header-only + template-only, so an unused
    // copy costs nothing. Inserted at the start so it precedes both the
    // linear-transform shim (which calls no key helper) and all functions.
    if (!keyShimAlreadyEmitted) {
      OpBuilder builder(ctx);
      builder.setInsertionPointToStart(module.getBody());
      emitc::VerbatimOp::create(builder, module.getLoc(), kKeyLookupShim);
    }

    // Erase the external `__heir_debug_*` declarations: ConvertDebugCall
    // already rewrote their call sites to emitc.call_opaque "__heir_debug", and
    // the upstream Cpp emitter cannot print an external (bodyless) func.func
    // (it mis-emits it as an empty zero-arg definition). Medusa's C++ prelude
    // declares + defines `__heir_debug`.
    SmallVector<func::FuncOp> debugDecls;
    getOperation()->walk([&](func::FuncOp fn) {
      if (fn.isExternal() && isDebugPort(fn.getName()))
        debugDecls.push_back(fn);
    });
    for (func::FuncOp fn : debugDecls) fn.erase();

    // Re-type payload-buffer args (lvalue/array, which a func cannot carry) to
    // C++ references, mutable iff written.
    llvm::StringSet<> refified;
    getOperation()->walk([&](func::FuncOp fn) {
      if (fn.isExternal()) return;
      Block& entry = fn.getBody().front();
      SmallVector<Type> inputs(fn.getFunctionType().getInputs().begin(),
                               fn.getFunctionType().getInputs().end());
      // Only the client-decrypt boundary func (whose float result the
      // hand-written harness declares as `float*`) gets its float-array result
      // flattened. Internal helpers (e.g. `_assign_layout`) also have
      // `bufferize.result` float outputs, but their callers are generated code
      // that passes multi-dim C arrays -- flattening those to `float*` would
      // break the in-module call (`no matching function`).
      bool clientDecrypt = fn->hasAttr("client.dec_func");
      bool changed = false;
      for (unsigned i = 0; i < inputs.size(); ++i) {
        bool written = isPayloadArgWritten(fn, i);
        Type ref = referenceArgType(ctx, inputs[i], written);
        if (!ref) {
          // Not a payload/handle arg. The client-decrypt float-array result
          // out-param is flattened to a `<elt>*` to match the harness.
          if (clientDecrypt && fn.getArgAttr(i, "bufferize.result")) {
            if (Type ptr = flattenFloatResultArg(entry.getArgument(i))) {
              inputs[i] = ptr;
              changed = true;
            }
          }
          continue;
        }
        inputs[i] = ref;
        entry.getArgument(i).setType(ref);
        changed = true;
      }
      if (changed) {
        fn.setType(
            FunctionType::get(ctx, inputs, fn.getFunctionType().getResults()));
        refified.insert(fn.getName());
      }
    });

    // Cross-function calls to ref-ified callees no longer type-check as
    // func.call; re-emit them as verbatim text calls (all are void after
    // buffer-results-to-out-params).
    SmallVector<func::CallOp> callsToRewrite;
    getOperation()->walk([&](func::CallOp call) {
      if (refified.contains(call.getCallee()) && call.getNumResults() == 0)
        callsToRewrite.push_back(call);
    });
    for (func::CallOp call : callsToRewrite) {
      OpBuilder b(call);
      std::string fmt =
          (call.getCallee() + "(" + placeholders(call.getNumOperands()) + ");")
              .str();
      VerbatimOp::create(b, call.getLoc(), fmt, call.getOperands());
      call.erase();
    }

    // Strip leftover tensor_ext.* boundary metadata (references the
    // unregistered tensor_ext dialect, which mlir-to-cpp can't parse).
    getOperation()->walk([](func::FuncOp fn) {
      auto stripTensorExt =
          [](DictionaryAttr d) -> std::optional<SmallVector<NamedAttribute>> {
        if (!d) return std::nullopt;
        SmallVector<NamedAttribute> kept;
        for (NamedAttribute a : d)
          if (!a.getName().strref().starts_with("tensor_ext."))
            kept.push_back(a);
        if (kept.size() == d.size()) return std::nullopt;
        return kept;
      };
      for (unsigned i = 0, e = fn.getNumArguments(); i < e; ++i)
        if (auto kept = stripTensorExt(fn.getArgAttrDict(i)))
          fn.setArgAttrs(i, *kept);
      for (unsigned i = 0, e = fn.getNumResults(); i < e; ++i)
        if (auto kept = stripTensorExt(fn.getResultAttrDict(i)))
          fn.setResultAttrs(i, *kept);
    });
  }
};

//===----------------------------------------------------------------------===//
// cheddar-externalize-weights pass
//===----------------------------------------------------------------------===//

// Write a large emitc.global weight initializer's raw float bytes to
// <data-dir>/<name>.bin and strip the initializer; emit a __load_constants()
// loader of verbatim `heir_load_f32(...)` calls. See the .td.
struct CheddarExternalizeWeights
    : impl::CheddarExternalizeWeightsBase<CheddarExternalizeWeights> {
  using CheddarExternalizeWeightsBase::CheddarExternalizeWeightsBase;

  void runOnOperation() override {
    if (dataDir.empty()) return;
    ModuleOp mod = cast<ModuleOp>(getOperation());
    MLIRContext* ctx = &getContext();

    if (std::error_code ec = llvm::sys::fs::create_directories(dataDir)) {
      mod.emitError() << "cheddar-externalize-weights: cannot create data-dir '"
                      << dataDir << "': " << ec.message();
      signalPassFailure();
      return;
    }

    SmallVector<std::pair<std::string, int64_t>> loaded;  // (name, numElements)
    // (name, numElements, valueLiteral) for large non-zero splats filled at
    // load time; all-zero splats need no entry (C++ static zero-init).
    SmallVector<std::tuple<std::string, int64_t, std::string>> splatFills;
    bool writeFailed = false;
    mod.walk([&](emitc::GlobalOp g) {
      if (writeFailed) return;
      Attribute init = g.getInitialValueAttr();
      if (!init) return;
      auto arrTy = dyn_cast<emitc::ArrayType>(g.getType());
      if (!arrTy || !isa<FloatType>(arrTy.getElementType())) return;
      int64_t n = 1;
      for (int64_t d : arrTy.getShape()) n *= d;

      ArrayRef<char> bytes;
      if (auto dr = dyn_cast<DenseResourceElementsAttr>(init)) {
        // torch-mlir stores large weights as resource blobs; externalize any.
        bytes = dr.getData();
        if (bytes.empty()) return;  // data not materialized; leave inline
      } else if (auto de = dyn_cast<DenseElementsAttr>(init)) {
        // Small dense constants emit fine inline.
        if (n <= threshold) return;
        // A *large* splat must never stay inline: heir-translate expands an
        // array initializer element-by-element, so a splat global (e.g. a
        // 512x65536 zero-padded ciphertext-width buffer) balloons the emitted
        // C++ by hundreds of MB and makes the host compiler OOM. Drop the
        // initializer instead -- an all-zero splat then relies on C++ static
        // zero-initialization (no data file), and a non-zero splat is filled
        // in __load_constants. Only genuine large non-splat weights get a blob.
        if (de.isSplat()) {
          APFloat sv = de.getSplatValue<APFloat>();
          g.removeInitialValueAttr();
          if (!sv.isZero()) {
            SmallString<32> sbuf;
            sv.toString(sbuf);
            splatFills.emplace_back(g.getSymName().str(), n, std::string(sbuf));
          }
          return;
        }
        bytes = de.getRawData();
      } else {
        return;
      }

      std::string path = dataDir + "/" + g.getSymName().str() + ".bin";
      std::error_code ec;
      llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_None);
      if (ec) {
        g.emitError() << "cannot write weight blob '" << path
                      << "': " << ec.message();
        writeFailed = true;
        return;
      }
      os.write(bytes.data(), bytes.size());
      os.close();

      g.removeInitialValueAttr();
      loaded.emplace_back(g.getSymName().str(), n);
    });
    if (writeFailed) {
      signalPassFailure();
      return;
    }
    if (loaded.empty() && splatFills.empty()) return;

    // Generate `void __load_constants()` that loads each blob into its global
    // and fills any large non-zero splat global.
    OpBuilder b(ctx);
    b.setInsertionPointToEnd(mod.getBody());
    Location loc = mod.getLoc();
    auto fn = func::FuncOp::create(b, loc, "__load_constants",
                                   FunctionType::get(ctx, {}, {}));
    fn.setPublic();
    b.setInsertionPointToStart(fn.addEntryBlock());
    for (auto& [name, n] : loaded) {
      std::string txt = "heir_load_f32(\"data/" + name + ".bin\", " +
                        "reinterpret_cast<float*>(" + name + "), " +
                        std::to_string(n) + ");";
      VerbatimOp::create(b, loc, txt, ValueRange{});
    }
    for (auto& [name, n, val] : splatFills) {
      std::string txt = "for (size_t __i = 0; __i < " + std::to_string(n) +
                        "; ++__i) reinterpret_cast<float*>(" + name +
                        ")[__i] = static_cast<float>(" + val + ");";
      VerbatimOp::create(b, loc, txt, ValueRange{});
    }
    func::ReturnOp::create(b, loc);
  }
};

}  // namespace

void registerCheddarConvertToEmitCInterface(DialectRegistry& registry) {
  registry.addExtension(
      +[](MLIRContext* ctx, cheddar::CheddarDialect* dialect) {
        dialect->addInterfaces<CheddarToEmitCDialectInterface>();
      });
  // Satisfy the func dialect's ConvertToEmitCPatternInterface promise without
  // pulling in the stock FuncToEmitC (the cheddar interface keeps func.func via
  // a structural conversion; see NoOpToEmitCInterface).
  registry.addExtension(+[](MLIRContext* ctx, func::FuncDialect* dialect) {
    dialect->addInterfaces<NoOpToEmitCInterface>();
  });
  // Likewise satisfy the memref dialect's promise with a no-op: the cheddar
  // interface OWNS memref->emitc lowering (it calls
  // populateMemRefToEmitCConversionPatterns itself plus higher-benefit custom
  // patterns, with its own memref type conversions). Letting the stock
  // MemRefToEmitC interface also register would add a competing set of
  // patterns/type-conversions to the shared converter whose visitation order
  // isn't controlled, producing irreconcilable ptr<->array casts on float
  // buffers. So `registerConvertMemRefToEmitCInterface` must NOT be called.
  registry.addExtension(+[](MLIRContext* ctx, mlir::memref::MemRefDialect* d) {
    d->addInterfaces<NoOpToEmitCInterface>();
  });
}

void registerCheddarToEmitCExternalModels(DialectRegistry& registry) {
  registry.addExtension(+[](MLIRContext* ctx, mlir::emitc::EmitCDialect*) {
    mlir::emitc::OpaqueType::attachInterface<EmitCOpaqueAsMemRefElement>(*ctx);
  });
}

}  // namespace mlir::heir
