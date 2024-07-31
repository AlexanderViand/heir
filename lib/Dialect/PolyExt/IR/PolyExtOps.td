#ifndef HEIR_LIB_DIALECT_POLYEXT_IR_POLYEXTOPS_TD_
#define HEIR_LIB_DIALECT_POLYEXT_IR_POLYEXTOPS_TD_

include "PolyExtDialect.td"
include "mlir/Dialect/Polynomial/IR/PolynomialTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


class PolyExt_Op<string mnemonic, list<Trait> traits = []> :
        Op<PolyExt_Dialect, mnemonic, traits> {
  let cppNamespace = "::mlir::heir::poly_ext";
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def PolyExt_CModSwitchOp : PolyExt_Op<"cmod_switch", traits = [Pure, ElementwiseMappable]> {
  let summary = "Move polynomial into a ring with different `cmod` by scaling polynomial coefficients proportionally";

  let description = [{
    `poly_ext.cmod_switch` "moves" polynomial $x$ from ring $Z/QZ[X]/(f(x))$ to the ring $Z/qZ[X]/(f(x))$.
    The move is done by scaling the polynomial coefficients by $q/Q$ and "rounding" so input
    and output coefficients are congruent modulo `congruence_modulus` $t$.
    The resulting polynomial $\tilde{x} \approx (q/Q)x$ and $\tilde{x} \equiv x \mod t$.
  }];
  let arguments = (ins
    PolynomialLike:$x,
    // TODO: make congruence_modulus optional with default value 1
    Builtin_IntegerAttr:$congruence_modulus
  );

  let results = (outs
    PolynomialLike:$output
  );

  let assemblyFormat = "$x attr-dict `:` type($x) `->` type($output)";

  let hasVerifier = 1;
}


// TODO: Not sure if this should live here or in PolyExt?
def PolyExt_GadgetProduct : PolyExt_Op<"gadget_product", traits = [Pure]> {
  let summary = "Noise-optimized product between two polynomials";
  let description = [{
    Perform a somehow noise-optimized product on two RLWE ciphertexts.
  }];

  let arguments = (ins
    PolynomialLike:$lhs,
    PolynomialLike:$rhs
  );
  let results = (outs PolynomialLike:$output);
}

def PolyExt_KSKDelta : PolyExt_Op<"ksk_delta", traits = [Pure]> {
  let summary = "Compute the delta of a keyswitching key";
  let description = [{
    Compute the delta of a keyswitching key.
  }];

  let arguments = (ins
    PolynomialLike:$x,
    //TODO: Do we want these to be attributes instead of, e.g., arith.constant?
    // Doing so might make it harder to see what's going on from textual IR, though
    AnyInteger:$t,
    AnyInteger:$m_t_inv
  );
  let results = (outs PolynomialLike:$output);
}

def PolyExt_DigitDecompoose : PolyExt_Op<"digit_decompose", traits = [Pure]> {
  let summary = "Decompose a polynomial into digits";
  let description = [{
    Decompose a polynomial into digits.
  }];

  let arguments = (ins
    PolynomialLike:$x
  );
  let results = (outs
    PolynomialLike:$output
  );
  let assemblyFormat = "$x attr-dict `:` type($x) `->` type($output)";
}

#endif  // HEIR_LIB_DIALECT_POLYEXT_IR_POLYEXTOPS_TD_
