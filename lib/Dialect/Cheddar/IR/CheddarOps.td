#ifndef LIB_DIALECT_CHEDDAR_IR_CHEDDAROPS_TD_
#define LIB_DIALECT_CHEDDAR_IR_CHEDDAROPS_TD_

include "CheddarDialect.td"
include "CheddarTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "lib/Dialect/HEIRInterfaces.td"

class Cheddar_Op<string mnemonic, list<Trait> traits = []> :
        Op<Cheddar_Dialect, mnemonic, traits> {
  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
}

// The CHEDDAR payload types are move-only and the C++ API is destination-passing
// (`ctx->Add(out, a, b)`). To match that, every payload-producing op is modeled
// as a destination-passing-style (DPS) op on builtin tensors: a scalar payload
// is a rank-0 `tensor<!cheddar.X>`, the trailing `$output` operand is the
// destination (init), and the result is tied 1:1 to it. The op is valid both
// in tensor form (one result, pre-bufferization) and in buffer form (operands
// are `memref<!cheddar.X>`, zero results, post one-shot-bufferize). The shared
// `getDpsInitsMutable` points at `$output`; `mad_unsafe` overrides it to point
// at its accumulator (true in-place).
class Cheddar_DpsOp<string mnemonic, list<Trait> traits = []> :
        Cheddar_Op<mnemonic, traits # [DestinationStyleOpInterface]> {
  let extraClassDeclaration = [{
    // The single `$output` init operand as a (length-1) MutableOperandRange.
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return ::mlir::MutableOperandRange(
          getOperation(), getOutputMutable().getOperandNumber(), 1);
    }
  }];
}

// Payload operands and results are constrained by their cheddar element type.
// `*Like` accepts a tensor or memref (valid pre- and post-bufferization);
// `*Result` is a variadic ranked tensor (present in tensor form, absent in
// buffer form).
def Cheddar_CtLike : TensorOrMemRef<[Cheddar_Ciphertext]>;
def Cheddar_PtLike : TensorOrMemRef<[Cheddar_Plaintext]>;
def Cheddar_ConstLike : TensorOrMemRef<[Cheddar_Constant]>;
def Cheddar_FloatLike : TensorOrMemRef<[AnyFloat, AnyComplex]>;

def Cheddar_CtResult : Variadic<RankedTensorOf<[Cheddar_Ciphertext]>>;
def Cheddar_PtResult : Variadic<RankedTensorOf<[Cheddar_Plaintext]>>;
def Cheddar_ConstResult : Variadic<RankedTensorOf<[Cheddar_Constant]>>;

//===----------------------------------------------------------------------===//
// Setup operations
//===----------------------------------------------------------------------===//

def Cheddar_CreateContextOp : Cheddar_Op<"create_context", [MemoryEffects<[MemWrite]>]> {
  let summary = "Create a CHEDDAR context from parameters";
  let description = [{
    Creates a CHEDDAR Context<word> from a Parameter object.
    The context is the main server-side computation engine.
  }];
  let arguments = (ins Cheddar_Parameter:$params);
  let results = (outs Cheddar_Context:$ctx);
}

def Cheddar_CreateUserInterfaceOp : Cheddar_Op<"create_user_interface", [MemoryEffects<[MemWrite]>]> {
  let summary = "Create a CHEDDAR UserInterface for key gen and encrypt/decrypt";
  let description = [{
    Creates a UserInterface<word> from a context. The UserInterface handles
    key generation, encryption, and decryption. Note: for test purposes only.
  }];
  let arguments = (ins Cheddar_AnyContext:$ctx);
  let results = (outs Cheddar_UserInterface:$ui);
}

def Cheddar_GetEncoderOp : Cheddar_Op<"get_encoder"> {
  let summary = "Get the encoder from a CHEDDAR context";
  let description = [{
    Returns a reference to the context's encoder (context->encoder_).
  }];
  let arguments = (ins Cheddar_AnyContext:$ctx);
  let results = (outs Cheddar_Encoder:$encoder);
}

def Cheddar_GetEvkMapOp : Cheddar_Op<"get_evk_map"> {
  let summary = "Get the evaluation key map from a UserInterface";
  let description = [{
    Returns the EvkMap from the UserInterface (ui.GetEvkMap()).
  }];
  let arguments = (ins Cheddar_UserInterface:$ui);
  let results = (outs Cheddar_EvkMap:$evkMap);
}

def Cheddar_GetMultKeyOp : Cheddar_Op<"get_mult_key"> {
  let summary = "Get the multiplication evaluation key";
  let description = [{
    Returns the multiplication key from the UserInterface
    (ui.GetMultiplicationKey()).
  }];
  let arguments = (ins Cheddar_UserInterface:$ui);
  let results = (outs Cheddar_EvalKey:$key);
}

def Cheddar_PrepareRotKeyOp : Cheddar_Op<"prepare_rot_key", [MemoryEffects<[MemWrite]>]> {
  let summary = "Generate a rotation key for a given distance";
  let description = [{
    Calls ui.PrepareRotationKey(distance, max_level) to generate a rotation key.
    Must be called before using rotation with that distance.
    The max_level parameter specifies the maximum ciphertext level at which
    this rotation key will be used.
  }];
  let arguments = (ins
    Cheddar_UserInterface:$ui,
    Builtin_IntegerAttr:$distance,
    Builtin_IntegerAttr:$maxLevel
  );
  let results = (outs);
}

//===----------------------------------------------------------------------===//
// Encode / Encrypt / Decrypt operations
//===----------------------------------------------------------------------===//

def Cheddar_EncodeOp : Cheddar_DpsOp<"encode", [PlaintextEncodeOpInterface]> {
  let summary = "Encode a message vector into a CHEDDAR plaintext";
  let description = [{
    Calls encoder.Encode(pt, level, scale, message). The message is a vector
    of complex numbers (or reals). `scale` is the C++ `double` value used by
    CHEDDAR's Encoder. `$output` is the destination plaintext.
  }];
  let arguments = (ins
    Cheddar_Encoder:$encoder,
    Cheddar_FloatLike:$message,
    Cheddar_PtLike:$output,
    Builtin_IntegerAttr:$level,
    F64Attr:$scale
  );
  let results = (outs Cheddar_PtResult:$result);
}

def Cheddar_EncodeConstantOp : Cheddar_DpsOp<"encode_constant"> {
  let summary = "Encode a scalar double into a CHEDDAR constant";
  let description = [{
    Calls encoder.EncodeConstant(constant, level, scale, number).
    The result is in RNS form for efficient ciphertext-scalar ops.
    `scale` is the C++ `double` value used by CHEDDAR's Encoder. `$output` is
    the destination constant.
  }];
  let arguments = (ins
    Cheddar_Encoder:$encoder,
    AnyFloat:$value,
    Cheddar_ConstLike:$output,
    Builtin_IntegerAttr:$level,
    F64Attr:$scale
  );
  let results = (outs Cheddar_ConstResult:$result);
}

def Cheddar_DecodeOp : Cheddar_Op<"decode", [DestinationStyleOpInterface]> {
  let summary = "Decode a CHEDDAR plaintext back to a message vector";
  let description = [{
    Calls encoder.Decode(message, pt). `value` is the destination buffer the
    decoded message is written into (the DPS init); the DPS interface verifier
    ties it 1:1 to the `decoded` result (same type).
  }];
  let arguments = (ins
    Cheddar_Encoder:$encoder,
    Cheddar_PtLike:$plaintext,
    Cheddar_FloatLike:$value
  );
  let results = (outs Variadic<AnyRankedTensor>:$decoded);
  let extraClassDeclaration = [{
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return ::mlir::MutableOperandRange(
          getOperation(), getValueMutable().getOperandNumber(), 1);
    }
  }];
}

def Cheddar_EncryptOp : Cheddar_DpsOp<"encrypt", [MemoryEffects<[MemWrite]>]> {
  let summary = "Encrypt a plaintext into a ciphertext";
  let description = [{
    Calls ui.Encrypt(ct, pt). Test-only operation. `$output` is the destination
    ciphertext.
  }];
  let arguments = (ins
    Cheddar_UserInterface:$ui,
    Cheddar_PtLike:$plaintext,
    Cheddar_CtLike:$output
  );
  let results = (outs Cheddar_CtResult:$result);
}

def Cheddar_DecryptOp : Cheddar_DpsOp<"decrypt", [MemoryEffects<[MemWrite]>]> {
  let summary = "Decrypt a ciphertext into a plaintext";
  let description = [{
    Calls ui.Decrypt(pt, ct). Test-only operation. `$output` is the destination
    plaintext.
  }];
  let arguments = (ins
    Cheddar_UserInterface:$ui,
    Cheddar_CtLike:$ciphertext,
    Cheddar_PtLike:$output
  );
  let results = (outs Cheddar_PtResult:$result);
}

//===----------------------------------------------------------------------===//
// Arithmetic operations
//===----------------------------------------------------------------------===//

class Cheddar_BinaryCtCtOp<string mnemonic, list<Trait> traits = []>
    : Cheddar_DpsOp<mnemonic, traits> {
  let arguments = (ins
    Cheddar_AnyContext:$ctx,
    Cheddar_CtLike:$lhs,
    Cheddar_CtLike:$rhs,
    Cheddar_CtLike:$output
  );
  let results = (outs Cheddar_CtResult:$result);
}

def Cheddar_AddOp : Cheddar_BinaryCtCtOp<"add"> {
  let summary = "Add two ciphertexts";
  let description = [{
    Calls context->Add(res, a, b).
  }];
}

def Cheddar_SubOp : Cheddar_BinaryCtCtOp<"sub"> {
  let summary = "Subtract two ciphertexts";
  let description = [{
    Calls context->Sub(res, a, b).
  }];
}

def Cheddar_MultOp : Cheddar_BinaryCtCtOp<"mult", [IncreasesMulDepthOpInterface]> {
  let summary = "Multiply two ciphertexts (tensor product, no relin/rescale)";
  let description = [{
    Calls context->Mult(res, a, b). Produces a degree-3 ciphertext.
    Does NOT include relinearization or rescaling.
  }];
}

def Cheddar_AddPlainOp : Cheddar_DpsOp<"add_plain"> {
  let summary = "Add a plaintext to a ciphertext";
  let arguments = (ins
    Cheddar_AnyContext:$ctx,
    Cheddar_CtLike:$ciphertext,
    Cheddar_PtLike:$plaintext,
    Cheddar_CtLike:$output
  );
  let results = (outs Cheddar_CtResult:$result);
}

def Cheddar_SubPlainOp : Cheddar_DpsOp<"sub_plain"> {
  let summary = "Subtract a plaintext from a ciphertext";
  let arguments = (ins
    Cheddar_AnyContext:$ctx,
    Cheddar_CtLike:$ciphertext,
    Cheddar_PtLike:$plaintext,
    Cheddar_CtLike:$output
  );
  let results = (outs Cheddar_CtResult:$result);
}

def Cheddar_MultPlainOp : Cheddar_DpsOp<"mult_plain"> {
  let summary = "Multiply a ciphertext by a plaintext (no rescale)";
  let arguments = (ins
    Cheddar_AnyContext:$ctx,
    Cheddar_CtLike:$ciphertext,
    Cheddar_PtLike:$plaintext,
    Cheddar_CtLike:$output
  );
  let results = (outs Cheddar_CtResult:$result);
}

def Cheddar_AddConstOp : Cheddar_DpsOp<"add_const"> {
  let summary = "Add a constant to a ciphertext";
  let arguments = (ins
    Cheddar_AnyContext:$ctx,
    Cheddar_CtLike:$ciphertext,
    Cheddar_ConstLike:$constant,
    Cheddar_CtLike:$output
  );
  let results = (outs Cheddar_CtResult:$result);
}

def Cheddar_MultConstOp : Cheddar_DpsOp<"mult_const"> {
  let summary = "Multiply a ciphertext by a constant (no rescale)";
  let arguments = (ins
    Cheddar_AnyContext:$ctx,
    Cheddar_CtLike:$ciphertext,
    Cheddar_ConstLike:$constant,
    Cheddar_CtLike:$output
  );
  let results = (outs Cheddar_CtResult:$result);
}

def Cheddar_NegOp : Cheddar_DpsOp<"neg"> {
  let summary = "Negate a ciphertext";
  let description = [{
    Calls context->Neg(res, a).
  }];
  let arguments = (ins
    Cheddar_AnyContext:$ctx,
    Cheddar_CtLike:$input,
    Cheddar_CtLike:$output
  );
  let results = (outs Cheddar_CtResult:$result);
}

def Cheddar_RescaleOp : Cheddar_DpsOp<"rescale"> {
  let summary = "Rescale a ciphertext (drop one level)";
  let description = [{
    Calls context->Rescale(res, a). Reduces the ciphertext level by 1.
  }];
  let arguments = (ins
    Cheddar_AnyContext:$ctx,
    Cheddar_CtLike:$input,
    Cheddar_CtLike:$output
  );
  let results = (outs Cheddar_CtResult:$result);
}

def Cheddar_LevelDownOp : Cheddar_DpsOp<"level_down"> {
  let summary = "Reduce ciphertext to a target level";
  let description = [{
    Calls context->LevelDown(res, a, target_level).
  }];
  let arguments = (ins
    Cheddar_AnyContext:$ctx,
    Cheddar_CtLike:$input,
    Cheddar_CtLike:$output,
    Builtin_IntegerAttr:$targetLevel
  );
  let results = (outs Cheddar_CtResult:$result);
}

//===----------------------------------------------------------------------===//
// Key-switching operations
//===----------------------------------------------------------------------===//

def Cheddar_RelinearizeOp : Cheddar_DpsOp<"relinearize"> {
  let summary = "Relinearize a ciphertext (without rescale)";
  let description = [{
    Calls context->Relinearize(res, a, mult_key).
  }];
  let arguments = (ins
    Cheddar_AnyContext:$ctx,
    Cheddar_CtLike:$input,
    Cheddar_EvalKey:$multKey,
    Cheddar_CtLike:$output
  );
  let results = (outs Cheddar_CtResult:$result);
}

def Cheddar_RelinearizeRescaleOp : Cheddar_DpsOp<"relinearize_rescale"> {
  let summary = "Fused relinearize + rescale";
  let description = [{
    Calls context->RelinearizeRescale(res, a, mult_key). Faster than
    separate Relinearize + Rescale.
  }];
  let arguments = (ins
    Cheddar_AnyContext:$ctx,
    Cheddar_CtLike:$input,
    Cheddar_EvalKey:$multKey,
    Cheddar_CtLike:$output
  );
  let results = (outs Cheddar_CtResult:$result);
}

//===----------------------------------------------------------------------===//
// Compound (fused) operations -- high-performance GPU kernels
//===----------------------------------------------------------------------===//

def Cheddar_HMultOp : Cheddar_DpsOp<"hmult", [IncreasesMulDepthOpInterface]> {
  let summary = "Fused multiply + relinearize (+ optional rescale)";
  let description = [{
    Calls context->HMult(res, a, b, mult_key, rescale).
    Single fused GPU kernel launch. The `rescale` attribute controls whether
    rescaling is included (default: true).
  }];
  let arguments = (ins
    Cheddar_AnyContext:$ctx,
    Cheddar_CtLike:$lhs,
    Cheddar_CtLike:$rhs,
    Cheddar_EvalKey:$multKey,
    Cheddar_CtLike:$output,
    DefaultValuedAttr<BoolAttr, "true">:$rescale
  );
  let results = (outs Cheddar_CtResult:$result);
}

def Cheddar_HRotOp : Cheddar_DpsOp<"hrot", [
  DeclareOpInterfaceMethods<RotationOpInterface>
]> {
  let summary = "Fused key-switch + rotation";
  let description = [{
    Calls context->HRot(res, a, rot_key, distance). The rotation key is looked
    up from the enclosing function's UserInterface using `distance`.
    Single fused GPU kernel. Supports both static and dynamic distances.
  }];
  let arguments = (ins
    Cheddar_AnyContext:$ctx,
    Cheddar_CtLike:$input,
    Cheddar_CtLike:$output,
    Optional<AnySignlessIntegerOrIndex>:$dynamic_distance,
    OptionalAttr<Builtin_IntegerAttr>:$static_distance
  );
  let results = (outs Cheddar_CtResult:$result);
  let hasVerifier = 1;
}

def Cheddar_HRotAddOp : Cheddar_DpsOp<"hrot_add", [
  DeclareOpInterfaceMethods<RotationOpInterface>
]> {
  let summary = "Fused rotation + addition";
  let description = [{
    Computes res = rotate(a, distance) + b in a single fused GPU kernel.
    Calls context->HRotAdd(res, a, b, rot_key, distance). The rotation key
    is looked up from the enclosing function's UserInterface using `distance`.
  }];
  let arguments = (ins
    Cheddar_AnyContext:$ctx,
    Cheddar_CtLike:$input,
    Cheddar_CtLike:$addend,
    Cheddar_CtLike:$output,
    Builtin_IntegerAttr:$distance
  );
  let results = (outs Cheddar_CtResult:$result);
}

def Cheddar_HConjOp : Cheddar_DpsOp<"hconj"> {
  let summary = "Fused key-switch + conjugation";
  let description = [{
    Calls context->HConj(res, a, conj_key). The conjugation key is looked
    up from the enclosing function's UserInterface.
  }];
  let arguments = (ins
    Cheddar_AnyContext:$ctx,
    Cheddar_CtLike:$input,
    Cheddar_CtLike:$output
  );
  let results = (outs Cheddar_CtResult:$result);
}

def Cheddar_HConjAddOp : Cheddar_DpsOp<"hconj_add"> {
  let summary = "Fused conjugation + addition";
  let description = [{
    Computes res = conj(a) + b in a single fused GPU kernel.
    Calls context->HConjAdd(res, a, b, conj_key). The conjugation key is
    looked up from the enclosing function's UserInterface.
  }];
  let arguments = (ins
    Cheddar_AnyContext:$ctx,
    Cheddar_CtLike:$input,
    Cheddar_CtLike:$addend,
    Cheddar_CtLike:$output
  );
  let results = (outs Cheddar_CtResult:$result);
}

def Cheddar_MadUnsafeOp : Cheddar_Op<"mad_unsafe", [DestinationStyleOpInterface]> {
  let summary = "Fused multiply-accumulate with constant (no rescale)";
  let description = [{
    Computes res += a * constant (in-place accumulation).
    Calls context->MadUnsafe(res, a, constant). The `$accumulator` operand is
    the in-place destination (the DPS init); the result is tied to it.

    Note: this op would typically be called `mac`,
    the current name reflects the spelling in Cheddar
    (`MadUnsafe`).
  }];
  let arguments = (ins
    Cheddar_AnyContext:$ctx,
    Cheddar_CtLike:$accumulator,
    Cheddar_CtLike:$input,
    Cheddar_ConstLike:$constant
  );
  let results = (outs Cheddar_CtResult:$result);
  let extraClassDeclaration = [{
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return ::mlir::MutableOperandRange(
          getOperation(), getAccumulatorMutable().getOperandNumber(), 1);
    }
  }];
}

//===----------------------------------------------------------------------===//
// Extension operations (bootstrapping, linear transforms, poly eval)
//===----------------------------------------------------------------------===//

def Cheddar_BootOp : Cheddar_DpsOp<"boot", [ResetsMulDepthOpInterface]> {
  let summary = "Bootstrap a ciphertext";
  let description = [{
    Calls boot_ctx->Boot(res, input, evk_map).
    Refreshes the ciphertext noise budget.

    Requires a `!cheddar.boot_context` (not a plain `!cheddar.context`): `Boot`
    is a BootContext method, so this op statically guarantees the context it
    runs on is a BootContext and the emitter needs no downcast.
  }];
  let arguments = (ins
    Cheddar_BootContext:$ctx,
    Cheddar_CtLike:$input,
    Cheddar_EvkMap:$evkMap,
    Cheddar_CtLike:$output
  );
  let results = (outs Cheddar_CtResult:$result);
}

def Cheddar_LinearTransformOp : Cheddar_DpsOp<"linear_transform", [
  DeclareOpInterfaceMethods<RotationOpInterface>
]> {
  let summary = "Apply a linear transform on a ciphertext";
  let description = [{
    Applies a matrix-vector product using CHEDDAR's LinearTransform extension
    with BSGS optimization and hoisting.

    The `diagonals` input is a 2D tensor where each row is a non-zero diagonal.
    The `diagonal_indices` attribute specifies which diagonal each row represents.
    The `level` attribute specifies the modulus level for the operation.
  }];
  let arguments = (ins
    Cheddar_AnyContext:$ctx,
    Cheddar_CtLike:$input,
    Cheddar_EvkMap:$evkMap,
    TensorOrMemRef<[AnyFloat]>:$diagonals,
    Cheddar_CtLike:$output,
    DenseI32ArrayAttr:$diagonal_indices,
    Builtin_IntegerAttr:$level,
    Builtin_IntegerAttr:$bs,
    Builtin_IntegerAttr:$gs
  );
  let results = (outs Cheddar_CtResult:$result);
  let hasVerifier = 1;
}

def Cheddar_EvalPolyOp : Cheddar_DpsOp<"eval_poly"> {
  let summary = "Evaluate a polynomial on a ciphertext";
  let description = [{
    Evaluates a polynomial (e.g., Chebyshev approximation) on an encrypted
    input using CHEDDAR's EvalPoly extension.
    `level` is the level the input ciphertext enters at; `outputLevel` is the
    level the result lands at after the polynomial's multiplicative depth.
  }];
  let arguments = (ins
    Cheddar_AnyContext:$ctx,
    Cheddar_CtLike:$input,
    Cheddar_EvkMap:$evkMap,
    Cheddar_CtLike:$output,
    F64ArrayAttr:$coefficients,
    Builtin_IntegerAttr:$level,
    Builtin_IntegerAttr:$outputLevel
  );
  let results = (outs Cheddar_CtResult:$result);
  let hasVerifier = 1;
}

#endif  // LIB_DIALECT_CHEDDAR_IR_CHEDDAROPS_TD_
