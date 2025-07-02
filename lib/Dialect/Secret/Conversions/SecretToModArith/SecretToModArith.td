#ifndef LIB_DIALECT_SECRET_CONVERSIONS_SECRETTOMODARITH_SECRETTOMODARITH_TD_
#define LIB_DIALECT_SECRET_CONVERSIONS_SECRETTOMODARITH_SECRETTOMODARITH_TD_

include "mlir/Pass/PassBase.td"

/// **Developer Documentation: SecretToModArith Pass**
///
/// This pass implements the lowering from the high-level `secret` dialect to the
/// `mod_arith` dialect, enabling plaintext evaluation of FHE algorithms for
/// testing, debugging, and prototyping purposes.
///
/// **Implementation Details:**
/// - Converts `secret.generic` operations to modular arithmetic operations
/// - Removes encryption/decryption overhead for faster plaintext execution
/// - Supports custom modulus selection for different arithmetic rings
/// - Handles floating-point encoding through scaling parameters
///
/// **Use Cases:**
/// - Algorithm prototyping before applying encryption
/// - Debugging FHE pipelines on plaintext data
/// - Performance comparison between encrypted and plaintext execution
/// - Testing correctness of secret computations
///
/// **Reusability Patterns:**
/// - Shares conversion infrastructure with other Secret-to-* passes
/// - Uses common pattern-based conversion framework
/// - Extensible for additional arithmetic operations
///
/// **Extension Points:**
/// - Add support for new modular arithmetic operations
/// - Implement custom encoding schemes for different data types
/// - Extend floating-point support with different scaling strategies
/// - Add support for multi-modulus arithmetic (RNS-style)
///
/// **Configuration Options:**
/// - Automatic modulus selection based on integer types when not specified
/// - Custom modulus for specific arithmetic requirements
/// - Log-scale parameter for floating-point to fixed-point conversion
///
/// **Dependencies:**
/// - Requires Arith and ModArith dialects
/// - Uses standard MLIR arithmetic operations as input
/// - Compatible with standard MLIR optimization passes
def SecretToModArith : Pass<"secret-to-mod-arith"> {
  let summary = "Lower `secret` to `mod-arith` dialect.";

  let description = [{
    This pass lowers an IR with `secret.generic` blocks containing arithmetic
    operations to operations on plaintexts using the `mod_arith` dialect.
    This is primarily used in the plaintext lowering pipeline, where operations
    are performed directly against plaintexts for development and testing.

    The pass assumes that the `secret.generic` regions have been distributed
    through arithmetic operations so that only one ciphertext operation appears
    per generic block. It also requires that `canonicalize` was run so that
    non-secret values used are removed from the `secret.generic`'s block
    arguments.

    **Technical Notes:**
    - Enables rapid prototyping and debugging of FHE algorithms
    - Supports automatic modulus selection based on integer types
    - Provides floating-point encoding through configurable scaling
    - Compatible with existing arithmetic optimization passes
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::heir::mod_arith::ModArithDialect",
  ];

  let options = [
    Option<"plaintextModulus", "modulus", "int64_t",
           /*default=*/"0", "Modulus to use for the mod-arith dialect. "
           "If not specified, the pass will use the natural modulus for that integer type">,
    Option<"logScale", "log-scale", "int64_t",
           /*default=*/"0", "Log base 2 of the scale for encoding floating points as ints.">,
  ];
}

#endif  // LIB_DIALECT_SECRET_CONVERSIONS_SECRETTOMODARITH_SECRETTOMODARITH_TD_
