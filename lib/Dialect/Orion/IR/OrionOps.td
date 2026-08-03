#ifndef LIB_DIALECT_ORION_IR_ORIONOPS_TD_
#define LIB_DIALECT_ORION_IR_ORIONOPS_TD_

include "lib/Dialect/HEIRInterfaces.td"
include "lib/Dialect/Orion/IR/OrionDialect.td"
include "lib/Dialect/LWE/IR/LWETypes.td"

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Orion_Op<string mnemonic, list<Trait> traits = []> :
        Op<Orion_Dialect, mnemonic, traits> {
  let cppNamespace = "::mlir::heir::orion";
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)" ;
}

def Orion_LinearTransformOp : Orion_Op<"linear_transform",
    [Pure, DeclareOpInterfaceMethods<ElementwiseByOperandOpInterface>]> {
  // Use a cleartext for the diagonals since they need to be rotated in the
  // BSGS implementation of linear transforms.
  // LWECiphertextLike so the torch-linalg path can produce it on a
  // ciphertext-semantic tensor<Nx!ct> (lowered from tensor_ext.rotate_and_reduce
  // at SecretToCKKS) as well as on a bare ciphertext (the orion frontend).
  let arguments = (ins
    LWECiphertextLike:$input,
    FloatLike:$diagonals,
    DenseI32ArrayAttr:$diagonal_indices,
    Builtin_IntegerAttr:$orion_level,
    Builtin_FloatAttr:$bsgs_ratio,
    Builtin_IntegerAttr:$slots
  );
  let results = (outs LWECiphertextLike:$result);
  let summary = "Apply a linear transformation to the input ciphertext using pre-computed diagonals";
  let description = [{
    This operation applies a linear transformation on a ciphertext using
    the provided float diagonals.

    The `diagonals` input is a tensor where each row represents one non-zero
    diagonal of the square matrix to evaluate.

    The `diagonal_indices` attribute specifies the index of each corresponding diagonal
    in `diagonals`. I.e., the first diagonal in the original matrix may have been zero
    and omitted, and as a result the first entry of `diagonals` corresponds to
    the diagonal with index `diagonal_indices[0]`, and so on.

    Semantics: the transform is square over `slots`; diagonal `d` multiplies
    the input rotated left by `d`, i.e.
    `out[i] += diagonals[k][i] * in[(i + diagonal_indices[k]) mod slots]`.
    Indices are interpreted modulo `slots` (canonical form `[0, slots)`).
    The result is at the input's level with scale `input_scale * pt_scale`
    (one plaintext-multiply depth); a `ckks.rescale` of the result is emitted
    explicitly at the scheme level. Backends whose linear-transform kernel
    rescales internally (CHEDDAR) absorb that rescale during lowering.
    The plaintext encoding scale of the diagonals is backend-chosen so that
    the post-rescale scale returns to the backend's canonical scale at the
    output level (Lattigo: the rescale prime q_level; CHEDDAR: GetScale(level)).

    The `orion_level` attribute specifies the modulus level at which the operation
    should be performed.

    The `bsgs_ratio` attribute is the *linear* target ratio of baby steps to
    giant steps for the baby-step/giant-step evaluation (the Orion frontend's
    convention, e.g. `2.0`). Backends whose API takes a log2 ratio (Lattigo's
    `LogBabyStepGiantStepRatio`) must convert.

    The `slots` attribute specifies the number of slots in the ciphertext.
  }];
}

def Orion_ChebyshevOp : Orion_Op<"chebyshev", [Pure, ElementwiseMappable]> {
  // LWECiphertextLike (+ ElementwiseMappable) so the op can be produced on a
  // ciphertext-semantic tensor<Nx!ct> (the torch-linalg path packs a layer
  // across N ciphertexts) and then scalarized to per-ciphertext orion.chebyshev
  // by ElementwiseToAffine before the Lattigo emitter (which handles one ct).
  let arguments = (ins
    LWECiphertextLike:$input,
    ArrayAttr:$coefficients,
    Builtin_FloatAttr:$domain_start,
    Builtin_FloatAttr:$domain_end,
    // Optional target output scale for the polynomial evaluation. Orion's
    // composite-sign chain sets the last polynomial's output scale to the
    // modulus `ql` at the output level (so the following ciphertext-multiply's
    // rescale is exact); the frontend conveys that ql here. Absent => the
    // lowering uses the default scale (2^logDefaultScale).
    OptionalAttr<Builtin_IntegerAttr>:$output_scale
  );
  let results = (outs LWECiphertextLike:$result);
  let summary = "Evaluates a Chebyshev polynomial on a ciphertext using pre-computed coefficients";
}

#endif  // LIB_DIALECT_ORION_IR_ORIONOPS_TD_
