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

Value addressOfFirstElement(OpBuilder& b, Location loc, Value array) {
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
  // memref<!cheddar.X> after bufferization, and strided float slices.
  tc.addConversion([&tc, ctx](MemRefType type) -> std::optional<Type> {
    std::string name = payloadTypeName(type.getElementType());
    if (!name.empty()) {
      Type payload = OpaqueType::get(ctx, name);
      if (type.getRank() == 0) return Type(LValueType::get(payload));
      if (type.hasStaticShape() && type.getRank() == 1)
        return Type(emitc::ArrayType::get(type.getShape(), payload));
      return Type();
    }
    // A strided/offset float slice feeding cheddar.encode -> a raw pointer.
    if (!type.getLayout().isIdentity()) {
      Type converted = tc.convertType(type.getElementType());
      if (converted && isa<FloatType>(converted) &&
          isa<StridedLayoutAttr>(type.getLayout()))
        return Type(emitc::PointerType::get(converted));
      return Type();
    }
    // Static identity-layout float memrefs -> a fixed-size C array (stack), so
    // memref.alloc becomes an `emitc.variable` (our ConvertAllocArray) rather
    // than a heap `aligned_alloc` (upstream MemRefToEmitC, whose `T*`->array
    // cast cannot be reconciled). Mirrors the array mapping for payload
    // memrefs.
    if (type.hasStaticShape() && type.getRank() > 0 &&
        !llvm::is_contained(type.getShape(), 0)) {
      Type elt = tc.convertType(type.getElementType());
      if (!elt) return Type();
      return Type(emitc::ArrayType::get(type.getShape(), elt));
    }
    return std::nullopt;
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
    auto arrayTy = dyn_cast<emitc::ArrayType>(dst.getType());
    if (!arrayTy || !isa<FloatType>(arrayTy.getElementType())) return failure();
    auto shape = arrayTy.getShape();
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

// memref.alloc of a rank-0 payload buffer -> a scalar local `T name;`.
struct ConvertAllocScalar : public OpConversionPattern<mlir::memref::AllocOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::AllocOp op, OpAdaptor /*adaptor*/,
      ConversionPatternRewriter& rewriter) const override {
    Type converted = getTypeConverter()->convertType(op.getType());
    if (!isa_and_present<emitc::LValueType>(converted)) return failure();
    rewriter.replaceOpWithNewOp<emitc::VariableOp>(
        op, converted, emitc::OpaqueAttr::get(rewriter.getContext(), ""));
    return success();
  }
};

// memref.alloc of a rank-1 payload buffer -> `T name[N];`.
struct ConvertAllocArray : public OpConversionPattern<mlir::memref::AllocOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::AllocOp op, OpAdaptor /*adaptor*/,
      ConversionPatternRewriter& rewriter) const override {
    Type converted = getTypeConverter()->convertType(op.getType());
    auto arrayTy = dyn_cast_or_null<emitc::ArrayType>(converted);
    if (!arrayTy || !isa<emitc::OpaqueType>(arrayTy.getElementType()))
      return failure();
    rewriter.replaceOpWithNewOp<emitc::VariableOp>(
        op, arrayTy, emitc::OpaqueAttr::get(rewriter.getContext(), ""));
    return success();
  }
};

// memref.dealloc of a payload buffer is a no-op (RAII handles cleanup).
struct EraseDealloc : public OpConversionPattern<mlir::memref::DeallocOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::DeallocOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Type elt;
    if (auto a = dyn_cast<emitc::ArrayType>(adaptor.getMemref().getType()))
      elt = a.getElementType();
    else if (auto l =
                 dyn_cast<emitc::LValueType>(adaptor.getMemref().getType()))
      elt = l.getValueType();
    if (!isa_and_present<emitc::OpaqueType>(elt)) return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

// memref.load on a rank-1 payload buffer -> subscript (an lvalue, fed to
// member_call operands directly).
struct ConvertLoadArray : public OpConversionPattern<mlir::memref::LoadOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::LoadOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto arrayTy = dyn_cast<emitc::ArrayType>(adaptor.getMemref().getType());
    if (!arrayTy || !isa<emitc::OpaqueType>(arrayTy.getElementType()))
      return failure();
    auto lvalT = emitc::LValueType::get(arrayTy.getElementType());
    rewriter.replaceOpWithNewOp<emitc::SubscriptOp>(
        op, lvalT, adaptor.getMemref(), adaptor.getIndices());
    return success();
  }
};

// memref.store of a payload into a rank-1 buffer -> `arr[i] = std::move(src);`.
struct ConvertStoreArray : public OpConversionPattern<mlir::memref::StoreOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::StoreOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto arrayTy = dyn_cast<emitc::ArrayType>(adaptor.getMemref().getType());
    if (!arrayTy || !isa<emitc::OpaqueType>(arrayTy.getElementType()))
      return failure();
    auto lvalT = emitc::LValueType::get(arrayTy.getElementType());
    auto sub =
        emitc::SubscriptOp::create(rewriter, op.getLoc(), lvalT,
                                   adaptor.getMemref(), adaptor.getIndices());
    VerbatimOp::create(rewriter, op.getLoc(), "{} = std::move({});",
                       ValueRange{sub.getResult(), adaptor.getValue()});
    rewriter.eraseOp(op);
    return success();
  }
};

// memref.copy of a payload buffer -> a move, not a (deleted) copy. These appear
// after buffer-results-to-out-params when a function returns a buffer it didn't
// allocate in place (e.g. a callee's out-param result, or an input buffer):
// the source is dead afterward, so moving is correct and a C++ copy (illegal
// for the move-only payload) is avoided.
//   * rank-0 (lvalue) scalar:  `dst = std::move(src);`
//   * rank-1 (array):          `std::move(std::begin(src), std::end(src),
//                                         std::begin(dst));`
struct ConvertCopyPayloadMove
    : public OpConversionPattern<mlir::memref::CopyOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::CopyOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value src = adaptor.getSource();
    Value tgt = adaptor.getTarget();
    auto isScalarPayload = [](Value v) {
      auto l = dyn_cast<emitc::LValueType>(v.getType());
      return l && isa<emitc::OpaqueType>(l.getValueType());
    };
    auto isArrayPayload = [](Value v) {
      auto a = dyn_cast<emitc::ArrayType>(v.getType());
      return a && isa<emitc::OpaqueType>(a.getElementType());
    };
    if (isScalarPayload(src) && isScalarPayload(tgt)) {
      VerbatimOp::create(rewriter, op.getLoc(), "{} = std::move({});",
                         ValueRange{tgt, src});
      rewriter.eraseOp(op);
      return success();
    }
    if (isArrayPayload(src) && isArrayPayload(tgt)) {
      VerbatimOp::create(
          rewriter, op.getLoc(),
          "std::move(std::begin({}), std::end({}), std::begin({}));",
          ValueRange{src, src, tgt});
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

// memref.subview producing a rank-0 scalar slice of a payload array (the
// rank-reducing extract_slice/insert_slice used to pull a single ciphertext out
// of / into a `tensor<1x!cheddar.X>` packing container) -> `base[o...]`, an
// lvalue subscript. Downstream cheddar ops and payload copies reference it by
// name, so no value is materialised.
struct ConvertSubViewPayloadScalar
    : public OpConversionPattern<mlir::memref::SubViewOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::SubViewOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value base = adaptor.getSource();
    auto arrayTy = dyn_cast<emitc::ArrayType>(base.getType());
    if (!arrayTy || !isa<emitc::OpaqueType>(arrayTy.getElementType()))
      return failure();
    // Only a fully-indexed, rank-reducing-to-scalar slice is supported (one
    // static offset per source dim, result rank 0).
    if (cast<MemRefType>(op.getType()).getRank() != 0) return failure();
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
    rewriter.replaceOpWithNewOp<emitc::SubscriptOp>(op, lvalT, base, idx);
    return success();
  }
};

// memref.subview producing a strided slice of a float buffer -> `&base[o...]`.
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

// Like upstream ConvertGlobal but tolerates an alignment attribute (bufferized
// constant globals carry `alignment = 64`; emitc.global has no alignas).
struct ConvertGlobalDropAlign
    : public OpConversionPattern<mlir::memref::GlobalOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::GlobalOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    if (!op.getType().hasStaticShape()) return failure();
    Type resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy) return failure();
    auto vis = SymbolTable::getSymbolVisibility(op);
    if (vis != SymbolTable::Visibility::Public &&
        vis != SymbolTable::Visibility::Private)
      return failure();
    bool staticSpecifier = vis == SymbolTable::Visibility::Private;
    Attribute initialValue = adaptor.getInitialValueAttr();
    if (isa_and_present<UnitAttr>(initialValue)) initialValue = {};
    rewriter.replaceOpWithNewOp<emitc::GlobalOp>(
        op, adaptor.getSymName(), resultTy, initialValue,
        /*externSpecifier=*/!staticSpecifier, staticSpecifier,
        adaptor.getConstant());
    return success();
  }
};

// memref.copy between float buffers -> an element-wise loop.
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
    auto begin = [](Value v) -> std::string {
      if (isa<emitc::PointerType>(v.getType())) return "({})";
      auto a = cast<emitc::ArrayType>(v.getType());
      std::string s = "(&{}";
      for (size_t i = 0; i < a.getShape().size(); ++i) s += "[0]";
      return s + ")";
    };
    std::string fmt = "for (size_t _i = 0; _i < " + std::to_string(n) +
                      "; ++_i) " + begin(tgt) + "[_i] = " + begin(src) +
                      "[_i];";
    VerbatimOp::create(rewriter, op.getLoc(), fmt, ValueRange{tgt, src});
    rewriter.eraseOp(op);
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

    // Payload memref ops + float-buffer ops (benefit 2 to win over the upstream
    // MemRefToEmitC patterns the memref interface contributes).
    patterns.add<ConvertAllocScalar, ConvertAllocArray, EraseDealloc,
                 ConvertLoadArray, ConvertStoreArray, ConvertCopyPayloadMove,
                 ConvertSubViewPayloadScalar, ConvertSubViewToPointer,
                 ConvertExpandShapeFloatCopy, ConvertGlobalDropAlign,
                 ConvertMemRefCopyFloat>(typeConverter, ctx, /*benefit=*/2);

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
    patterns.add<OutParamDpsPattern<cheddar::LevelDownOp>>(
        typeConverter, ctx, "LevelDown",
        std::function<std::string(cheddar::LevelDownOp)>(
            [](cheddar::LevelDownOp op) {
              return intLit(op.getTargetLevelAttr());
            }));
    patterns.add<OutParamDpsPattern<cheddar::HMultOp>>(
        typeConverter, ctx, "HMult",
        std::function<std::string(cheddar::HMultOp)>([](cheddar::HMultOp op) {
          return op.getRescale() ? std::string("true") : std::string("false");
        }));
  }
};

//===----------------------------------------------------------------------===//
// cheddar-emitc-boundary pass
//===----------------------------------------------------------------------===//

// Build the C++ reference type for a converted payload-buffer arg:
//   lvalue<opaque T> -> `T&` / `const T&`
//   emitc.array<NxT> -> `std::array<T, N>&` / `const std::array<T, N>&`
Type referenceArgType(MLIRContext* ctx, Type converted, bool written) {
  std::string base;
  if (auto l = dyn_cast<emitc::LValueType>(converted)) {
    if (auto o = dyn_cast<emitc::OpaqueType>(l.getValueType()))
      base = o.getValue().str();
  } else if (auto a = dyn_cast<emitc::ArrayType>(converted)) {
    if (auto o = dyn_cast<emitc::OpaqueType>(a.getElementType());
        o && a.getShape().size() == 1)
      base = ("std::array<" + o.getValue() + ", " +
              std::to_string(a.getShape()[0]) + ">")
                 .str();
  }
  if (base.empty()) return {};
  std::string name = written ? (base + "&") : ("const " + base + "&");
  return OpaqueType::get(ctx, name);
}

// A value is "written" if used as the destination (out) of an emitted call:
// member_call_opaque places the dest at operand 1 (after the receiver), a
// call_opaque shim (RunLinearTransform/RunEvalPoly) at operand 0, or an
// assignment verbatim (`{} = ...`) at operand 0.
bool valueWrittenAsDest(Value v) {
  for (OpOperand& use : v.getUses()) {
    Operation* owner = use.getOwner();
    unsigned idx = use.getOperandNumber();
    if (isa<MemberCallOpaqueOp>(owner) && idx == 1) return true;
    if (isa<CallOpaqueOp>(owner) && idx == 0) return true;
    if (auto vb = dyn_cast<VerbatimOp>(owner))
      if (idx == 0 && vb.getValue().contains("=")) return true;
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
      bool changed = false;
      for (unsigned i = 0; i < inputs.size(); ++i) {
        bool written = isPayloadArgWritten(fn, i);
        Type ref = referenceArgType(ctx, inputs[i], written);
        if (!ref) continue;
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
}

void registerCheddarToEmitCExternalModels(DialectRegistry& registry) {
  registry.addExtension(+[](MLIRContext* ctx, mlir::emitc::EmitCDialect*) {
    mlir::emitc::OpaqueType::attachInterface<EmitCOpaqueAsMemRefElement>(*ctx);
  });
}

}  // namespace mlir::heir
