#include "lib/Conversions/CheddarToEmitC/CheddarToEmitC.h"

#include <cstdio>
#include <functional>
#include <optional>
#include <string>

#include "lib/Dialect/Cheddar/IR/CheddarDialect.h"
#include "lib/Dialect/Cheddar/IR/CheddarOps.h"
#include "lib/Dialect/Cheddar/IR/CheddarTypes.h"
#include "lib/Dialect/Preprocessing/IR/PreprocessingOps.h"
#include "lib/Utils/TargetUtils.h"
#include "llvm/include/llvm/ADT/DenseSet.h"         // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"      // from @llvm-project
#include "llvm/include/llvm/ADT/StringMap.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/StringSet.h"        // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/ConvertToEmitC/ToEmitCInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Conversion/MemRefToEmitC/MemRefToEmitC.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/EmitC/IR/EmitC.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/Transforms/FuncConversions.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/Utils/MemRefUtils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"         // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"       // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"       // from @llvm-project
#include "mlir/include/mlir/IR/SymbolTable.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"              // from @llvm-project
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

constexpr StringLiteral kDestinationOperandAttr = "cheddar.destination_operand";

template <typename OpTy>
OpTy markDestination(OpTy op, unsigned operandNumber) {
  op->setAttr(
      kDestinationOperandAttr,
      IntegerAttr::get(IntegerType::get(op.getContext(), 64), operandNumber));
  return op;
}

std::optional<unsigned> getDestinationOperand(Operation* op) {
  auto attr = op->getAttrOfType<IntegerAttr>(kDestinationOperandAttr);
  if (!attr) return std::nullopt;
  return static_cast<unsigned>(attr.getInt());
}

// The CHEDDAR payload C++ type name for a cheddar element type, or "" if `t`
// isn't a (move-only) cheddar payload type.
std::string payloadTypeName(Type t) {
  if (isa<cheddar::CiphertextType>(t)) return "Ciphertext<word>";
  if (isa<cheddar::PlaintextType>(t)) return "Plaintext<word>";
  if (isa<cheddar::ConstantType>(t)) return "Constant<word>";
  if (isa<cheddar::EvalKeyType>(t)) return "EvaluationKey<word>";
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

int64_t numElements(ArrayRef<int64_t> shape) {
  int64_t result = 1;
  for (int64_t dim : shape) result *= dim;
  return result;
}

SmallVector<Value> flattenIndices(OpBuilder& builder, Location loc,
                                  ArrayRef<int64_t> shape, ValueRange indices) {
  auto sizeT = emitc::SizeTType::get(builder.getContext());
  if (indices.empty())
    return {emitc::LiteralOp::create(builder, loc, sizeT, "0")};
  if (indices.size() == 1) return {indices.front()};

  Value flat = indices.front();
  for (size_t i = 1; i < indices.size(); ++i) {
    Value dim =
        emitc::LiteralOp::create(builder, loc, sizeT, std::to_string(shape[i]));
    flat = emitc::MulOp::create(builder, loc, TypeRange{flat.getType()}, flat,
                                dim);
    flat = emitc::AddOp::create(builder, loc, TypeRange{flat.getType()}, flat,
                                indices[i]);
  }
  return {flat};
}

void ensureCleartextResourceInclude(Operation* op, OpBuilder& builder) {
  ModuleOp module = op->getParentOfType<ModuleOp>();
  constexpr StringLiteral header = "lib/Runtime/CleartextResource.h";
  for (auto include : module.getOps<emitc::IncludeOp>())
    if (include.getInclude() == header) return;
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  emitc::IncludeOp::create(builder, op->getLoc(), header,
                           /*isStandardInclude=*/false);
}

void ensureStandardInclude(Operation* op, OpBuilder& builder,
                           StringRef header) {
  ModuleOp module = op->getParentOfType<ModuleOp>();
  for (auto include : module.getOps<emitc::IncludeOp>())
    if (include.getIsStandardInclude() && include.getInclude() == header)
      return;
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  emitc::IncludeOp::create(builder, op->getLoc(), header,
                           /*isStandardInclude=*/true);
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
  markDestination(
      MemberCallOpaqueOp::create(b, loc, /*resultTypes=*/TypeRange{}, receiver,
                                 b.getStringAttr(method), argsAttr,
                                 /*template_args=*/ArrayAttr{}, argOperands),
      1);
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
  tc.addConversion(
      [ctx](IndexType) -> Type { return emitc::SizeTType::get(ctx); });
  // Bufferized buffers split by element kind:
  //  * move-only cheddar PAYLOAD (ciphertext/plaintext/constant/key): a place
  //    carried as `lvalue<opaque>` -- rank-0 -> `T name;`; rank>=1 -> a
  //    (nested) `std::array` (`std::array<T,N> name;`). std::array (not a C
  //    array) so it binds to the `std::array<T,N>&` function-boundary refs and
  //    the harness; subscripting works via the patched emitc.subscript
  //    (lvalue<opaque> base).
  //  * Primitive cleartext buffers: a flat pointer. Storage is represented
  //    independently: memref.alloca lowers to a local C-array owner and its
  //    first-element pointer, memref.alloc lowers to a heap pointer, and
  //    memref.global lowers to static C-array storage whose address is taken by
  //    memref.get_global. This keeps every memref use on one representation
  //    without conflating an EmitC array value with heap storage.
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
    std::string payloadName = payloadTypeName(eltType);
    bool payload = !payloadName.empty();
    if (!payload && !isa<FloatType, IntegerType>(eltType)) return std::nullopt;
    if (type.getRank() == 0) {
      if (payload)
        return Type(LValueType::get(OpaqueType::get(ctx, payloadName)));
      return Type(emitc::PointerType::get(eltType));
    }
    if (payload) {
      if (!type.hasStaticShape() || llvm::is_contained(type.getShape(), 0))
        return Type();
      if (!memref::isStaticShapeAndContiguousRowMajor(type)) return Type();
      return Type(LValueType::get(
          OpaqueType::get(ctx, stdArrayName(type.getShape(), payloadName))));
    }
    if (!type.hasStaticShape() || llvm::is_contained(type.getShape(), 0) ||
        !memref::isStaticShapeAndContiguousRowMajor(type))
      return Type();
    return Type(emitc::PointerType::get(eltType));
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

// cheddar.encode: fill a std::vector<Complex> from the float message buffer,
// then encode at the requested logarithmic scale, or CHEDDAR's canonical scale
// for the level when no explicit scale is present.
struct ConvertEncode : public OpConversionPattern<cheddar::EncodeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::EncodeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value out = adaptor.getOutput();
    std::string lvl = std::to_string(op.getLevelAttr().getInt());
    Value msg = adaptor.getMessage();
    auto messageType = dyn_cast<ShapedType>(op.getMessage().getType());
    if (!messageType || !messageType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "encode requires a static message shape");
    int64_t n = numElements(messageType.getShape());
    // Flatten both a whole (possibly multidimensional) C array and a subview
    // pointer to the first scalar element before constructing the vector.
    Value begin = addressOfFirstElement(rewriter, op.getLoc(), msg);
    Value vec =
        VariableOp::create(rewriter, op.getLoc(),
                           LValueType::get(OpaqueType::get(
                               rewriter.getContext(), "std::vector<Complex>")),
                           OpaqueAttr::get(rewriter.getContext(), ""));
    VerbatimOp::create(
        rewriter, op.getLoc(),
        "{} = std::vector<Complex>({}, {} + " + std::to_string(n) + ");",
        ValueRange{vec, begin, begin});
    // TODO(#2364): Use scale from op once HEIR can do precise scale tracking.
    std::string scale = "{}.GetScale(" + lvl + ")";
    SmallVector<Value> operands{adaptor.getEncoder(), out,
                                adaptor.getEncoder()};
    operands.push_back(vec);
    markDestination(
        VerbatimOp::create(rewriter, op.getLoc(),
                           "{}.Encode({}, " + lvl + ", " + scale + ", {});",
                           operands),
        1);
    rewriter.eraseOp(op);
    return success();
  }
};

// cheddar.encode_constant: encode at the same canonical per-level scale.
struct ConvertEncodeConstant
    : public OpConversionPattern<cheddar::EncodeConstantOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cheddar::EncodeConstantOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    std::string lvl = intLit(op.getLevelAttr());
    std::string fmt =
        "{}.EncodeConstant({}, " + lvl + ", {}.GetScale(" + lvl + "), {});";
    markDestination(VerbatimOp::create(
                        rewriter, op.getLoc(), fmt,
                        ValueRange{adaptor.getEncoder(), adaptor.getOutput(),
                                   adaptor.getEncoder(), adaptor.getValue()}),
                    1);
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
    auto memTy = dyn_cast<MemRefType>(op.getValue().getType());
    if (!memTy || !memTy.hasStaticShape() ||
        !isa<FloatType>(memTy.getElementType()))
      return failure();
    auto pointerType = dyn_cast<emitc::PointerType>(dst.getType());
    if (!pointerType || pointerType.getPointee() != memTy.getElementType())
      return failure();
    auto shape = memTy.getShape();
    int64_t n = numElements(shape);
    auto* ctx = rewriter.getContext();
    Value vec = VariableOp::create(
        rewriter, op.getLoc(),
        LValueType::get(OpaqueType::get(ctx, "std::vector<Complex>")),
        OpaqueAttr::get(ctx, ""));
    markDestination(
        VerbatimOp::create(
            rewriter, op.getLoc(), "{}.Decode({}, {});",
            ValueRange{adaptor.getEncoder(), vec, adaptor.getPlaintext()}),
        1);
    markDestination(
        VerbatimOp::create(rewriter, op.getLoc(),
                           "for (size_t _i = 0; _i < " + std::to_string(n) +
                               "; ++_i) {}[_i] = {}.at(_i).real();",
                           ValueRange{dst, vec}),
        0);
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
      markDestination(
          VerbatimOp::create(
              rewriter, op.getLoc(),
              "{}->HRot({}, {}, {}->GetRotationKey(" + d + "), " + d + ");",
              ValueRange{adaptor.getCtx(), out, adaptor.getInput(), ui}),
          1);
    } else {
      Value dyn = adaptor.getDynamicDistance();
      markDestination(
          VerbatimOp::create(rewriter, op.getLoc(),
                             "{}->HRot({}, {}, {}->GetRotationKey({}), {});",
                             ValueRange{adaptor.getCtx(), out,
                                        adaptor.getInput(), ui, dyn, dyn}),
          1);
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
    markDestination(
        VerbatimOp::create(
            rewriter, op.getLoc(),
            "{}->HRotAdd({}, {}, {}, {}->GetRotationKey(" + d + "), " + d +
                ");",
            ValueRange{adaptor.getCtx(), adaptor.getOutput(),
                       adaptor.getInput(), adaptor.getAddend(), ui}),
        1);
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
    markDestination(
        VerbatimOp::create(rewriter, op.getLoc(),
                           "{}->HConj({}, {}, {}->GetConjugationKey());",
                           ValueRange{adaptor.getCtx(), adaptor.getOutput(),
                                      adaptor.getInput(), ui}),
        1);
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
    markDestination(
        VerbatimOp::create(
            rewriter, op.getLoc(),
            "{}->HConjAdd({}, {}, {}, {}->GetConjugationKey());",
            ValueRange{adaptor.getCtx(), adaptor.getOutput(),
                       adaptor.getInput(), adaptor.getAddend(), ui}),
        1);
    rewriter.eraseOp(op);
    return success();
  }
};

// cheddar.eval_poly -> the real EvalPoly<word> class (no free
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
//     EvalPoly<word> ep({coeffs}, lvl, is, ts, /*chebyshev=*/true);
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
    int64_t levelConsumption = op.getLevelConsumptionAttr().getInt();

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
    emit("EvalPoly<word> _ep(" + floatArrayLit(op.getCoefficientsAttr()) +
             ", _ep_lvl, _ep_is, _ep_ts, true);",
         {});
    emit("_ep.Compile(_ep_cp);", {});
    StringRef evaluate =
        "_ep.Evaluate(_ep_cp, {}, {}, {}.GetMultiplicationKey());";
    markDestination(
        VerbatimOp::create(rewriter, loc, rewriter.getStringAttr(evaluate),
                           ValueRange{out, in, evk}),
        0);
    emit("}", {});

    rewriter.eraseOp(op);
    return success();
  }
};

// A `__heir_debug_*` call (from --lwe-add-debug-port, re-shaped by LWEToCheddar
// to (Encoder, UserInterface, Ciphertext)) -> a free C++ call to an
// externally-defined `__heir_debug(encoder, ui, ct, "name", "metadata")`. The
// debug name/metadata travel as `debug.name`/`debug.metadata` dialect attrs;
// they are baked into the call as trailing string-literal args. The external
// `func.func` declaration is erased by the cheddar-emitc-boundary pass because
// the upstream Cpp emitter cannot print an external func.func declaration.
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

// memref.alloc of a Cheddar payload buffer -> a local C++ owning handle. The
// handle itself lives in automatic storage, while the Cheddar value owns its
// device allocation. Primitive cleartext memref.alloc operations deliberately
// fall through to upstream's heap-allocation lowering.
struct ConvertAllocLocal : public OpConversionPattern<mlir::memref::AllocOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::AllocOp op, OpAdaptor /*adaptor*/,
      ConversionPatternRewriter& rewriter) const override {
    Type converted = getTypeConverter()->convertType(op.getType());
    if (!isa_and_present<emitc::LValueType>(converted)) return failure();
    auto variable = emitc::VariableOp::create(
        rewriter, op.getLoc(), converted,
        emitc::OpaqueAttr::get(rewriter.getContext(), ""));
    rewriter.replaceOp(op, variable);
    return success();
  }
};

// File loading fills the storage selected by bufferization through its
// uniformly pointer-converted destination.
struct ConvertLoadResource
    : public OpConversionPattern<preprocessing::LoadResourceOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      preprocessing::LoadResourceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto memrefType = dyn_cast<MemRefType>(op.getDestination().getType());
    if (!memrefType) return failure();
    auto pointerType =
        dyn_cast<emitc::PointerType>(adaptor.getDestination().getType());
    if (!pointerType || pointerType.getPointee() != memrefType.getElementType())
      return failure();
    ensureCleartextResourceInclude(op, rewriter);
    std::string path = "\"";
    for (char c : op.getPath()) {
      if (c == '"' || c == '\\') path += '\\';
      path += c;
    }
    path += '"';
    auto args = rewriter.getArrayAttr(
        {emitc::OpaqueAttr::get(rewriter.getContext(), path),
         rewriter.getIndexAttr(0),
         emitc::OpaqueAttr::get(
             rewriter.getContext(),
             std::to_string(numElements(memrefType.getShape())))});
    emitc::CallOpaqueOp::create(
        rewriter, op.getLoc(), TypeRange{}, "heir::loadResource",
        ValueRange{adaptor.getDestination()}, args,
        rewriter.getArrayAttr({TypeAttr::get(memrefType.getElementType())}));
    rewriter.eraseOp(op);
    return success();
  }
};

// Primitive cleartext memrefs use one flat pointer representation regardless
// of whether their storage came from alloc, alloca, a global, or a subview.
struct ConvertLoadPointer : public OpConversionPattern<mlir::memref::LoadOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::LoadOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto pointerType =
        dyn_cast<emitc::PointerType>(adaptor.getMemref().getType());
    if (!pointerType) return failure();

    Type elementType = op.getMemRefType().getElementType();
    if (pointerType.getPointee() != elementType ||
        !isa<FloatType, IntegerType>(elementType))
      return failure();

    SmallVector<Value> indices =
        flattenIndices(rewriter, op.getLoc(), op.getMemRefType().getShape(),
                       adaptor.getIndices());
    auto subscript = emitc::SubscriptOp::create(
        rewriter, op.getLoc(), emitc::LValueType::get(elementType),
        adaptor.getMemref(), indices);
    auto loaded = emitc::LoadOp::create(rewriter, op.getLoc(), elementType,
                                        subscript.getResult());
    rewriter.replaceOp(op, loaded);
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
    // A surviving primitive pointer came from memref.alloc. Stack-promoted
    // resources are memref.alloca and therefore have no dealloc operation.
    if (isa<emitc::PointerType>(memTy)) {
      ensureStandardInclude(op, rewriter, "cstdlib");
      emitc::CallOpaqueOp::create(rewriter, op.getLoc(), TypeRange{}, "free",
                                  ValueRange{adaptor.getMemref()});
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

// memref.load on a payload std::array buffer (lvalue<opaque>) -> `base[i...]`
// via emitc.subscript. The subscript stays an lvalue because Cheddar payloads
// are move-only and consumed by reference.
struct ConvertLoadArray : public OpConversionPattern<mlir::memref::LoadOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::LoadOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Type baseTy = adaptor.getMemref().getType();
    bool isPayloadBuf = isa<emitc::LValueType>(baseTy);
    if (!isPayloadBuf) return failure();
    Type elt =
        getTypeConverter()->convertType(op.getMemRefType().getElementType());
    if (!elt) return failure();
    // Rank-0 payload memref: a rank-0 `memref<!cheddar.X>` converts to the
    // element lvalue itself (lvalue<opaque>), so a no-index load IS that lvalue
    // -- no subscript. Mirrors the indexed payload case which yields an lvalue
    // (used by preprocessing storage of a single plaintext slot). Non-payload
    // rank-0 loads fall through to the stock memref->emitc patterns.
    if (adaptor.getIndices().empty()) {
      if (isa<emitc::OpaqueType>(elt)) {
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
    auto loaded =
        emitc::LoadOp::create(rewriter, op.getLoc(), elt, sub.getResult());
    rewriter.replaceOp(op, loaded);
    return success();
  }
};

// The dialect-conversion materialization for an opaque memref.store's value
// operand may turn an `lvalue<opaque<T>>` producer into the `opaque<T>` value
// form via an unrealized_conversion_cast. Look through that cast so the
// assignment uses the lvalue directly.
static Value unwrapSingleUnrealizedCast(Value v) {
  if (auto cast = v.getDefiningOp<mlir::UnrealizedConversionCastOp>();
      cast && cast.getInputs().size() == 1 &&
      isa<emitc::LValueType>(cast.getInputs()[0].getType()))
    return cast.getInputs()[0];
  return v;
}

// Diagnose move-only copies before dialect conversion tries to materialize
// converted operands. An OpConversionPattern may never reach matchAndRewrite
// when such a materialization is impossible, which would hide the ownership
// error behind a generic "failed to legalize" diagnostic.
struct RejectMoveOnlyStore : public OpRewritePattern<mlir::memref::StoreOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::memref::StoreOp op,
                                PatternRewriter& /*rewriter*/) const override {
    Type elementType = op.getMemRefType().getElementType();
    if (payloadTypeName(elementType).empty() &&
        !isa<cheddar::UserInterfaceType>(elementType))
      return failure();
    return op.emitOpError(
        "copying a move-only Cheddar payload with memref.store is invalid");
  }
};

struct HandleMoveOnlyCopy : public OpRewritePattern<mlir::memref::CopyOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::memref::CopyOp op,
                                PatternRewriter& rewriter) const override {
    auto sourceType = cast<MemRefType>(op.getSource().getType());
    Type elementType = sourceType.getElementType();
    if (op.getSource() == op.getTarget()) {
      rewriter.eraseOp(op);
      return success();
    }
    if (isa<cheddar::CiphertextType>(elementType)) return failure();
    if (payloadTypeName(elementType).empty() &&
        !isa<cheddar::UserInterfaceType>(elementType))
      return failure();
    return op.emitOpError(
        "copying a move-only Cheddar value with memref.copy is invalid");
  }
};

// Copies that survive bufferization's standard alias folding have true copy
// semantics. Lower ciphertext copies through CHEDDAR's deep-copy API instead
// of reinterpreting them as C++ assignment or ownership transfer.
struct ConvertCiphertextCopy
    : public OpConversionPattern<mlir::memref::CopyOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::CopyOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto sourceType = cast<MemRefType>(op.getSource().getType());
    if (!isa<cheddar::CiphertextType>(sourceType.getElementType()))
      return failure();
    if (op.getSource() == op.getTarget()) {
      rewriter.eraseOp(op);
      return success();
    }

    Value context;
    if (auto function = op->getParentOfType<func::FuncOp>()) {
      for (BlockArgument argument : function.getArguments()) {
        auto pointer = dyn_cast<emitc::PointerType>(argument.getType());
        auto opaque = pointer
                          ? dyn_cast<emitc::OpaqueType>(pointer.getPointee())
                          : emitc::OpaqueType();
        if (opaque && (opaque.getValue() == "Context<word>" ||
                       opaque.getValue() == "BootContext<word>")) {
          context = argument;
          break;
        }
      }
    }
    if (!context)
      return op.emitOpError(
          "cannot deep-copy a Cheddar ciphertext without a context argument");

    emitOutParamCall(rewriter, op.getLoc(), context, "Copy",
                     adaptor.getTarget(), ValueRange{adaptor.getSource()});
    rewriter.eraseOp(op);
    return success();
  }
};

// memref.store retains value-copy semantics. Primitive cleartext buffers use a
// flat pointer; CHEDDAR's move-only payloads must never reach this conversion
// and their producers must write directly into destination buffers.
struct ConvertStoreArray : public OpConversionPattern<mlir::memref::StoreOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::StoreOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Type baseTy = adaptor.getMemref().getType();
    bool isPointer = isa<emitc::PointerType>(baseTy);
    if (!isPointer && !isa<emitc::LValueType>(baseTy)) return failure();
    Type elt =
        getTypeConverter()->convertType(op.getMemRefType().getElementType());
    if (!elt) return failure();
    // Rank-0 payload memref: store directly into the element lvalue. A rank-0
    // primitive pointer still subscripts element zero through flattenIndices.
    if (!isPointer && adaptor.getIndices().empty()) {
      if (isa<emitc::LValueType>(baseTy) && isa<emitc::OpaqueType>(elt)) {
        markDestination(
            VerbatimOp::create(
                rewriter, op.getLoc(), "{} = {};",
                ValueRange{adaptor.getMemref(),
                           unwrapSingleUnrealizedCast(adaptor.getValue())}),
            0);
        rewriter.eraseOp(op);
        return success();
      }
      return failure();
    }
    SmallVector<Value> indices =
        isPointer ? flattenIndices(rewriter, op.getLoc(),
                                   op.getMemRefType().getShape(),
                                   adaptor.getIndices())
                  : SmallVector<Value>(adaptor.getIndices().begin(),
                                       adaptor.getIndices().end());
    auto sub = emitc::SubscriptOp::create(rewriter, op.getLoc(),
                                          emitc::LValueType::get(elt),
                                          adaptor.getMemref(), indices);
    if (isa<emitc::OpaqueType>(elt)) {
      markDestination(
          VerbatimOp::create(
              rewriter, op.getLoc(), "{} = {};",
              ValueRange{sub.getResult(),
                         unwrapSingleUnrealizedCast(adaptor.getValue())}),
          0);
    } else {
      emitc::AssignOp::create(rewriter, op.getLoc(), sub.getResult(),
                              adaptor.getValue());
    }
    rewriter.eraseOp(op);
    return success();
  }
};

// memref.copy also retains value-copy semantics. It cannot be reinterpreted as
// an ownership transfer merely because a source happens to be dead.
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
    markDestination(VerbatimOp::create(rewriter, op.getLoc(), "{} = {};",
                                       ValueRange{tgt, src}),
                    0);
    rewriter.eraseOp(op);
    return success();
  }
};

// memref.subview slicing a std::array buffer (lvalue<opaque>) -> `base[o...]`,
// an lvalue subscript. The rank-reducing extract/insert slices used to pull a
// single ciphertext out of a `tensor<1x!cheddar.X>` packing container, or a row
// out of a `<1x1024xf32>` buffer, drop leading (size-1) dims: emit a subscript
// indexing those dimensions by their offsets, yielding the converted inner
// buffer or scalar. A nested C++ array cannot represent rank reduction of a
// non-leading dimension, so reject that layout explicitly.
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
    llvm::SmallBitVector droppedDims = op.getDroppedDims();
    for (int64_t i = 0; i < srcRank; ++i)
      if (droppedDims.test(i) != (i < dropped))
        return rewriter.notifyMatchFailure(
            op, "payload subview may drop only leading dimensions");
    // Subscript the dropped (leading) dims by their offsets. Offsets may be
    // static (a literal index) or dynamic (a loop induction variable, e.g. from
    // a `tensor.insert_slice` in an scf.for) -- emitc.subscript accepts dynamic
    // index operands, so thread the converted dynamic offset value through.
    auto mixedOffsets = op.getMixedOffsets();
    if (static_cast<int64_t>(mixedOffsets.size()) != srcRank) return failure();
    auto staticSizes = op.getStaticSizes();
    auto staticStrides = op.getStaticStrides();
    ArrayRef<int64_t> sourceShape = op.getSourceType().getShape();
    for (int64_t i = dropped; i < srcRank; ++i) {
      if (ShapedType::isDynamic(staticSizes[i]) ||
          staticSizes[i] != sourceShape[i] || staticStrides[i] != 1)
        return rewriter.notifyMatchFailure(
            op, "payload subview must retain a full contiguous suffix");
      auto offset = mixedOffsets[i];
      auto attr = dyn_cast<Attribute>(offset);
      if (!attr || cast<IntegerAttr>(attr).getInt() != 0)
        return rewriter.notifyMatchFailure(
            op, "payload subview must have zero retained-dimension offsets");
    }
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
    auto subscript =
        emitc::SubscriptOp::create(rewriter, op.getLoc(), resultTy, base, idx);
    rewriter.replaceOp(op, subscript);
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

// memref.subview producing a contiguous cleartext slice -> pointer arithmetic
// using the source memref's actual static strides.
struct ConvertSubViewToPointer
    : public OpConversionPattern<mlir::memref::SubViewOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::SubViewOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value base = adaptor.getSource();
    auto pointerTy = dyn_cast<emitc::PointerType>(base.getType());
    Type elementType = pointerTy ? pointerTy.getPointee() : Type{};
    if (!isa_and_present<FloatType, IntegerType>(elementType)) return failure();
    auto resultType = cast<MemRefType>(op.getType());
    if (!memref::isStaticShapeAndContiguousRowMajor(resultType))
      return rewriter.notifyMatchFailure(op,
                                         "float subview must be contiguous");
    auto offsets = op.getStaticOffsets();
    auto sourceType = op.getSourceType();
    if (static_cast<int64_t>(offsets.size()) != sourceType.getRank())
      return failure();
    for (int64_t o : offsets)
      if (ShapedType::isDynamic(o)) return failure();
    SmallVector<int64_t> sourceStrides;
    int64_t sourceOffset;
    if (failed(sourceType.getStridesAndOffset(sourceStrides, sourceOffset)) ||
        sourceOffset != 0 ||
        llvm::is_contained(sourceStrides, ShapedType::kDynamic))
      return failure();
    int64_t linearOffset = 0;
    for (int64_t i = 0; i < sourceType.getRank(); ++i)
      linearOffset += offsets[i] * sourceStrides[i];
    if (linearOffset == 0) {
      rewriter.replaceOp(op, base);
      return success();
    }
    auto literal = emitc::LiteralOp::create(
        rewriter, op.getLoc(),
        emitc::OpaqueType::get(getContext(), "std::ptrdiff_t"),
        std::to_string(linearOffset));
    auto offset = emitc::AddOp::create(
        rewriter, op.getLoc(), TypeRange{base.getType()}, base, literal);
    rewriter.replaceOp(op, offset.getResult());
    return success();
  }
};

// memref.copy between flat primitive pointers -> an element-wise loop (the
// move-only payload copy is handled by ConvertCopy).
struct ConvertMemRefCopyPrimitive
    : public OpConversionPattern<mlir::memref::CopyOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::CopyOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value src = adaptor.getSource();
    Value tgt = adaptor.getTarget();
    auto primitiveOperand = [](Value v) -> bool {
      if (auto p = dyn_cast<emitc::PointerType>(v.getType()))
        return isa<FloatType, IntegerType>(p.getPointee());
      return false;
    };
    if (!primitiveOperand(src) || !primitiveOperand(tgt) ||
        src.getType() != tgt.getType())
      return failure();
    auto sourceType = cast<MemRefType>(op.getSource().getType());
    if (!sourceType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "primitive copy requires static shape");
    int64_t n = numElements(sourceType.getShape());
    // Index both buffers through flat `<elt>*` SSA pointers so a result
    // out-param target (later flattened to `float*` at the boundary) stays
    // valid; the printed C++ is unchanged for ordinary C-array copies.
    Value tgtPtr = addressOfFirstElement(rewriter, op.getLoc(), tgt);
    Value srcPtr = addressOfFirstElement(rewriter, op.getLoc(), src);
    std::string fmt = "for (size_t _i = 0; _i < " + std::to_string(n) +
                      "; ++_i) ({})[_i] = ({})[_i];";
    markDestination(VerbatimOp::create(rewriter, op.getLoc(), fmt,
                                       ValueRange{tgtPtr, srcPtr}),
                    0);
    rewriter.eraseOp(op);
    return success();
  }
};

bool isPositiveZeroSplat(Attribute attr) {
  auto elements = dyn_cast_if_present<ElementsAttr>(attr);
  if (!elements || !elements.isSplat()) return false;

  Attribute splat = elements.getSplatValue<Attribute>();
  if (auto floatAttr = dyn_cast<FloatAttr>(splat))
    return floatAttr.getValue().isPosZero();
  if (auto intAttr = dyn_cast<IntegerAttr>(splat))
    return intAttr.getValue().isZero();
  return false;
}

// Like upstream ConvertGlobal but tolerates an alignment attribute (bufferized
// constants carry `alignment = 64`; emitc.global has no alignas). A cleartext
// global owns C-array storage even though every converted memref handle is a
// flat pointer.
struct ConvertGlobalDropAlign
    : public OpConversionPattern<mlir::memref::GlobalOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::GlobalOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    MemRefType type = op.getType();
    if (!type.hasStaticShape() ||
        !isa<FloatType, IntegerType>(type.getElementType()))
      return failure();
    Type storageType = type.getRank() == 0
                           ? type.getElementType()
                           : Type(emitc::ArrayType::get(type.getShape(),
                                                        type.getElementType()));
    auto vis = SymbolTable::getSymbolVisibility(op);
    if (vis != SymbolTable::Visibility::Public &&
        vis != SymbolTable::Visibility::Private)
      return failure();
    bool staticSpecifier = vis == SymbolTable::Visibility::Private;
    Attribute initialValue = adaptor.getInitialValueAttr();
    if (auto resource =
            dyn_cast_if_present<DenseResourceElementsAttr>(initialValue)) {
      initialValue = DenseElementsAttr::getFromRawBuffer(resource.getType(),
                                                         resource.getData());
    }
    if (type.getRank() == 0) {
      if (!op.getInitialValue()) return failure();
      auto elements = dyn_cast<ElementsAttr>(*op.getInitialValue());
      if (!elements) return failure();
      initialValue = elements.getSplatValue<Attribute>();
    }
    if (isa_and_present<UnitAttr>(initialValue)) initialValue = {};
    // A private global has static storage duration, so omitting a positive-zero
    // splat initializer zero-initializes the entire array without making the
    // C++ emitter print every element. Keep other splats, including -0.0.
    if (staticSpecifier && isPositiveZeroSplat(initialValue)) initialValue = {};
    // Emit the global non-const even when the source memref is `constant`.
    // Cleartext memref handles currently lower to non-const pointers, so a
    // static const array could not supply the pointer expected by its uses.
    // These globals are machine-generated and never mutated; preserving
    // source-level constness can be added once the memref type conversion can
    // represent a pointer-to-const element.
    auto global = emitc::GlobalOp::create(
        rewriter, op.getLoc(), adaptor.getSymName(), storageType, initialValue,
        /*externSpecifier=*/!staticSpecifier, staticSpecifier,
        /*constSpecifier=*/false);
    rewriter.replaceOp(op, global);
    return success();
  }
};

// memref.get_global returns the same flat pointer used by alloc/alloca-backed
// cleartext memrefs. The symbol itself remains an emitc.array (or a scalar for
// rank zero), so take its address explicitly at the ownership boundary.
struct ConvertGetGlobalPointer
    : public OpConversionPattern<mlir::memref::GetGlobalOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::memref::GetGlobalOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    MemRefType type = op.getType();
    if (!type.hasStaticShape() ||
        !isa<FloatType, IntegerType>(type.getElementType()))
      return failure();
    auto pointerType = dyn_cast_if_present<emitc::PointerType>(
        getTypeConverter()->convertType(type));
    if (!pointerType || pointerType.getPointee() != type.getElementType())
      return failure();
    if (type.getRank() == 0) {
      auto lvalueType = emitc::LValueType::get(type.getElementType());
      Value global = emitc::GetGlobalOp::create(
          rewriter, op.getLoc(), lvalueType, adaptor.getNameAttr());
      rewriter.replaceOp(op, emitc::AddressOfOp::create(rewriter, op.getLoc(),
                                                        pointerType, global));
      return success();
    }
    auto arrayType =
        emitc::ArrayType::get(type.getShape(), type.getElementType());
    Value global = emitc::GetGlobalOp::create(rewriter, op.getLoc(), arrayType,
                                              adaptor.getNameAttr());
    rewriter.replaceOp(op,
                       addressOfFirstElement(rewriter, op.getLoc(), global));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertToEmitC dialect interface
//===----------------------------------------------------------------------===//

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
    target.addIllegalOp<preprocessing::LoadResourceOp>();

    // Pull in the stock MemRefToEmitC patterns for the plain float ops
    // (alloc/load/store/global) at default benefit; the custom payload/float
    // patterns below at benefit 2 win where they apply. (This helper does not
    // touch the type converter -- the memref type conversions live in
    // addCheddarEmitCTypeConversions.)
    mlir::populateMemRefToEmitCConversionPatterns(patterns, typeConverter);

    // Payload memref ops + float-buffer ops (benefit 2 to win over the stock
    // MemRefToEmitC patterns added just above).
    patterns.add<
        ConvertAllocLocal, EraseDealloc, ConvertLoadPointer, ConvertLoadArray,
        ConvertStoreArray, ConvertCopy, ConvertMemRefCopyPrimitive,
        ConvertSubViewSubscript, ConvertSubViewToPointer, ConvertPayloadCast,
        ConvertGlobalDropAlign, ConvertGetGlobalPointer, ConvertLoadResource>(
        typeConverter, ctx, /*benefit=*/2);
    patterns.add<RejectMoveOnlyStore, HandleMoveOnlyCopy>(ctx,
                                                          /*benefit=*/3);
    patterns.add<ConvertCiphertextCopy>(typeConverter, ctx, /*benefit=*/3);

    patterns
        .add<ConvertEncode, ConvertEncodeConstant, ConvertDecode, ConvertHRot,
             ConvertHRotAdd, ConvertHConj, ConvertHConjAdd, ConvertEvalPoly>(
            typeConverter, ctx);

    auto addDps = [&](StringRef name, auto opTag,
                      std::function<std::string(decltype(opTag))> extra =
                          nullptr) {
      using Op = decltype(opTag);
      patterns.add<OutParamDpsPattern<Op>>(typeConverter, ctx, name, extra);
    };
    addDps("Copy", cheddar::CopyOp{});
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

// Determine whether an emitted operation writes this value. Conversion
// patterns mark their exact destination operand; call sites inherit the
// corresponding callee-argument classification. Subscripts and casts preserve
// the underlying storage identity for this purpose.
bool valueWrittenAsDest(
    Value root,
    const llvm::StringMap<SmallVector<bool>>& writtenFunctionArguments) {
  SmallVector<Value> worklist{root};
  llvm::DenseSet<Value> visited;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second) continue;
    for (OpOperand& use : value.getUses()) {
      Operation* owner = use.getOwner();
      unsigned operandNumber = use.getOperandNumber();
      if (getDestinationOperand(owner) == operandNumber) return true;
      if (auto call = dyn_cast<func::CallOp>(owner)) {
        auto it = writtenFunctionArguments.find(call.getCallee());
        if (it != writtenFunctionArguments.end() &&
            operandNumber < it->second.size() && it->second[operandNumber])
          return true;
      }
      if (auto subscript = dyn_cast<emitc::SubscriptOp>(owner);
          subscript && operandNumber == 0)
        worklist.push_back(subscript.getResult());
      if (auto cast = dyn_cast<UnrealizedConversionCastOp>(owner);
          cast && cast->getNumResults() == 1)
        worklist.push_back(cast->getResult(0));
    }
  }
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

    // Erase the external `__heir_debug_*` declarations: ConvertDebugCall
    // already rewrote their call sites to emitc.call_opaque "__heir_debug", and
    // the upstream Cpp emitter cannot print an external (bodyless) func.func
    // (it mis-emits it as an empty zero-arg definition).
    SmallVector<func::FuncOp> debugDecls;
    getOperation()->walk([&](func::FuncOp fn) {
      if (fn.isExternal() && isDebugPort(fn.getName()))
        debugDecls.push_back(fn);
    });
    for (func::FuncOp fn : debugDecls) fn.erase();

    // Compute mutable arguments to a fixed point over structured func.call
    // edges. This preserves const-correctness through helper functions without
    // trying to recover write intent from emitted C++ text.
    llvm::StringMap<SmallVector<bool>> writtenFunctionArguments;
    getOperation()->walk([&](func::FuncOp fn) {
      if (fn.isExternal()) return;
      SmallVector<bool> written(fn.getNumArguments(), false);
      for (unsigned i = 0; i < fn.getNumArguments(); ++i)
        written[i] = static_cast<bool>(fn.getArgAttr(i, "bufferize.result"));
      writtenFunctionArguments[fn.getName()] = std::move(written);
    });
    bool changedWritten;
    do {
      changedWritten = false;
      getOperation()->walk([&](func::FuncOp fn) {
        if (fn.isExternal()) return;
        SmallVector<bool>& written = writtenFunctionArguments[fn.getName()];
        for (unsigned i = 0; i < fn.getNumArguments(); ++i) {
          if (written[i]) continue;
          if (valueWrittenAsDest(fn.getArgument(i), writtenFunctionArguments)) {
            written[i] = true;
            changedWritten = true;
          }
        }
      });
    } while (changedWritten);

    // Re-type payload-buffer lvalues (which a func cannot carry) to C++
    // references, mutable iff written. Primitive memrefs are already pointers.
    llvm::StringSet<> refified;
    getOperation()->walk([&](func::FuncOp fn) {
      if (fn.isExternal()) return;
      Block& entry = fn.getBody().front();
      SmallVector<Type> inputs(fn.getFunctionType().getInputs().begin(),
                               fn.getFunctionType().getInputs().end());
      bool changed = false;
      for (unsigned i = 0; i < inputs.size(); ++i) {
        bool written = writtenFunctionArguments[fn.getName()][i];
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
    // func.call. Keep their operands structured in emitc.call_opaque instead
    // of formatting the entire call as untyped verbatim text.
    SmallVector<func::CallOp> callsToRewrite;
    getOperation()->walk([&](func::CallOp call) {
      if (refified.contains(call.getCallee())) callsToRewrite.push_back(call);
    });
    for (func::CallOp call : callsToRewrite) {
      OpBuilder b(call);
      auto rewritten = CallOpaqueOp::create(
          b, call.getLoc(), call.getResultTypes(),
          b.getStringAttr(call.getCallee()), /*args=*/ArrayAttr{},
          /*templateArgs=*/ArrayAttr{}, call.getOperands());
      call.replaceAllUsesWith(rewritten.getResults());
      call.erase();
    }

    // Destination markers are compiler-internal ABI metadata. The C++
    // translation no longer needs them after argument mutability is fixed.
    getOperation()->walk(
        [](Operation* op) { op->removeAttr(kDestinationOperandAttr); });

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
}

void registerCheddarToEmitCExternalModels(DialectRegistry& registry) {
  registry.addExtension(+[](MLIRContext* ctx, mlir::emitc::EmitCDialect*) {
    mlir::emitc::OpaqueType::attachInterface<EmitCOpaqueAsMemRefElement>(*ctx);
  });
}

}  // namespace mlir::heir
