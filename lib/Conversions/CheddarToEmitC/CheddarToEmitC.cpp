#include "lib/Conversions/CheddarToEmitC/CheddarToEmitC.h"

#include <cstdio>
#include <functional>
#include <optional>
#include <string>

#include "lib/Dialect/Cheddar/IR/CheddarDialect.h"
#include "lib/Dialect/Cheddar/IR/CheddarOps.h"
#include "lib/Dialect/Cheddar/IR/CheddarTypes.h"
#include "lib/Utils/ConversionUtils.h"
#include "llvm/include/llvm/ADT/DenseSet.h"     // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"    // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "llvm/include/llvm/ADT/StringSet.h"    // from @llvm-project
#include "mlir/include/mlir/Conversion/ConvertToEmitC/ToEmitCInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/MemRefToEmitC/MemRefToEmitC.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/EmitC/IR/EmitC.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/Transforms/FuncConversions.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
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

// True if an emitc opaque value type names a (move-only) cheddar payload, i.e.
// the buffer's elements are move-only. Used to choose move vs copy semantics.
bool opaqueNamesPayload(StringRef name) {
  return name.contains("Ciphertext<word>") ||
         name.contains("Plaintext<word>") || name.contains("Constant<word>") ||
         name.contains("EvaluationKey<word>");
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

// The enclosing function's UserInterface argument (looked up by converted
// type); used by the rotation ops, which look up `ui->GetRotationKey(d)`
// inline.
Value findUi(Operation* op, const TypeConverter& tc) {
  Type uiType =
      tc.convertType(cheddar::UserInterfaceType::get(op->getContext()));
  auto r = getContextualArgFromFunc(op, uiType);
  if (failed(r)) return Value{};
  return r.value();
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
    return OpaqueType::get(ctx, "Parameter");
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
    if (isa<cheddar::GetEvkMapOp, cheddar::GetMultKeyOp, cheddar::GetEncoderOp,
            cheddar::CreateUserInterfaceOp>(op)) {
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

// CreateContext: static factory, `T x = T::Create(args);`.
struct ConvertCreateContext
    : public OpConversionPattern<cheddar::CreateContextOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::CreateContextOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Type t = typeConverter->convertType(op.getResult().getType());
    auto call = CallOpaqueOp::create(
        rewriter, op.getLoc(), TypeRange{t},
        rewriter.getStringAttr("Context<word>::Create"),
        ValueRange{adaptor.getParams()}, ArrayAttr{}, ArrayAttr{});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

struct ConvertPrepareRotKey
    : public OpConversionPattern<cheddar::PrepareRotKeyOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::PrepareRotKeyOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    std::string extra =
        intLit(op.getDistanceAttr()) + ", " + intLit(op.getMaxLevelAttr());
    VerbatimOp::create(rewriter, op.getLoc(),
                       "{}->PrepareRotationKey(" + extra + ");",
                       ValueRange{adaptor.getUi()});
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
    Value ui = findUi(op, *typeConverter);
    if (!ui)
      return op.emitOpError("enclosing function is missing UserInterface arg");
    Value out = adaptor.getOutput();
    if (auto sd = op.getStaticDistanceAttr()) {
      std::string d = intLit(sd);
      VerbatimOp::create(
          rewriter, op.getLoc(),
          "{}->HRot({}, {}, {}->GetRotationKey(" + d + "), " + d + ");",
          ValueRange{adaptor.getCtx(), out, adaptor.getInput(), ui});
    } else {
      Value dyn = adaptor.getDynamicDistance();
      VerbatimOp::create(
          rewriter, op.getLoc(),
          "{}->HRot({}, {}, {}->GetRotationKey({}), {});",
          ValueRange{adaptor.getCtx(), out, adaptor.getInput(), ui, dyn, dyn});
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
    Value ui = findUi(op, *typeConverter);
    if (!ui)
      return op.emitOpError("enclosing function is missing UserInterface arg");
    std::string d = intLit(op.getDistanceAttr());
    VerbatimOp::create(
        rewriter, op.getLoc(),
        "{}->HRotAdd({}, {}, {}, {}->GetRotationKey(" + d + "), " + d + ");",
        ValueRange{adaptor.getCtx(), adaptor.getOutput(), adaptor.getInput(),
                   adaptor.getAddend(), ui});
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertHConj : public OpConversionPattern<cheddar::HConjOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::HConjOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value ui = findUi(op, *typeConverter);
    if (!ui)
      return op.emitOpError("enclosing function is missing UserInterface arg");
    VerbatimOp::create(rewriter, op.getLoc(),
                       "{}->HConj({}, {}, {}->GetConjugationKey());",
                       ValueRange{adaptor.getCtx(), adaptor.getOutput(),
                                  adaptor.getInput(), ui});
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertHConjAdd : public OpConversionPattern<cheddar::HConjAddOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::HConjAddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value ui = findUi(op, *typeConverter);
    if (!ui)
      return op.emitOpError("enclosing function is missing UserInterface arg");
    VerbatimOp::create(rewriter, op.getLoc(),
                       "{}->HConjAdd({}, {}, {}, {}->GetConjugationKey());",
                       ValueRange{adaptor.getCtx(), adaptor.getOutput(),
                                  adaptor.getInput(), adaptor.getAddend(), ui});
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

// cheddar.eval_poly -> RunEvalPoly<word>(out, ctx, in, evk_map,
// {coefficients}, level, outputLevel).
struct ConvertEvalPoly : public OpConversionPattern<cheddar::EvalPolyOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::EvalPolyOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    SmallVector<Value> operands{adaptor.getOutput(), adaptor.getCtx(),
                                adaptor.getInput(), adaptor.getEvkMap()};
    std::string trailing = floatArrayLit(op.getCoefficientsAttr()) + ", " +
                           intLit(op.getLevelAttr()) + ", " +
                           intLit(op.getOutputLevelAttr());
    SmallVector<Attribute> args;
    for (size_t i = 0; i < operands.size(); ++i)
      args.push_back(rewriter.getIndexAttr(i));
    args.push_back(emitc::OpaqueAttr::get(rewriter.getContext(), trailing));
    SmallVector<Attribute> templateArgs{
        emitc::OpaqueAttr::get(rewriter.getContext(), "word")};
    CallOpaqueOp::create(rewriter, op.getLoc(), TypeRange{},
                         rewriter.getStringAttr("RunEvalPoly"), operands,
                         rewriter.getArrayAttr(args),
                         rewriter.getArrayAttr(templateArgs));
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

// memref.dealloc of a cheddar buffer (lvalue<opaque>) is a no-op -- the
// scope-bound emitc.variable is freed automatically. Erase it.
struct EraseDealloc : public OpConversionPattern<mlir::memref::DeallocOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::DeallocOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto l = dyn_cast<emitc::LValueType>(adaptor.getMemref().getType());
    if (!l || !isa<emitc::OpaqueType>(l.getValueType())) return failure();
    rewriter.eraseOp(op);
    return success();
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
    if (adaptor.getIndices().empty()) return failure();
    Type elt =
        getTypeConverter()->convertType(op.getMemRefType().getElementType());
    if (!elt) return failure();
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
    if (adaptor.getIndices().empty()) return failure();
    Type elt =
        getTypeConverter()->convertType(op.getMemRefType().getElementType());
    if (!elt) return failure();
    auto sub = emitc::SubscriptOp::create(
        rewriter, op.getLoc(), emitc::LValueType::get(elt), adaptor.getMemref(),
        adaptor.getIndices());
    if (isa<emitc::OpaqueType>(elt)) {
      VerbatimOp::create(rewriter, op.getLoc(), "{} = std::move({});",
                         ValueRange{sub.getResult(), adaptor.getValue()});
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
          return typeConverter.isLegal(op);
        });

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
                 ConvertExpandShapeFloatCopy, ConvertGlobalDropAlign>(
        typeConverter, ctx, /*benefit=*/2);

    patterns.add<ConvertCreateContext, ConvertPrepareRotKey, ConvertEncode,
                 ConvertEncodeConstant, ConvertDecode, ConvertHRot,
                 ConvertHRotAdd, ConvertHConj, ConvertHConjAdd,
                 ConvertLinearTransform, ConvertEvalPoly>(typeConverter, ctx);

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

struct CheddarToEmitCPass
    : public impl::CheddarToEmitCBase<CheddarToEmitCPass> {
  using CheddarToEmitCBase::CheddarToEmitCBase;

  void runOnOperation() override {
    auto* ctx = &getContext();
    if (diagnoseUnsupportedGetters(getOperation())) {
      signalPassFailure();
      return;
    }

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
