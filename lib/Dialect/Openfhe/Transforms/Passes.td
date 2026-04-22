#ifndef LIB_DIALECT_OPENFHE_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_OPENFHE_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def AdjustCKKSSchemeParam : Pass<"openfhe-adjust-ckks-scheme-param"> {
  let summary = "Adjust CKKS scheme parameters for OpenFHE-native depth semantics";
  let description = [{
    This pass runs before OpenFHE lowering and updates `ckks.schemeParam` when
    OpenFHE-native CKKS semantics require a longer modulus chain than the
    current module attribute provides.

    It models OpenFHE's ciphertext noise-scale-degree law on the already
    managed CKKS/Orion IR, then regenerates the scheme parameters if a longer
    chain is required. The later `openfhe-configure-crypto-context` pass then
    reads the resulting scheme parameters directly instead of rediscovering
    structured-op depth requirements from OpenFHE IR.

    The exact per-level scale schedule still depends on the selected OpenFHE
    scaling technique. This pass handles only the shared chain-length
    recurrence. Exact scale planning stays in the CKKS management pipeline.
  }];
  let options = [
    Option<"scalingTechnique", "scaling-technique", "std::string",
           /*default=*/"\"\"",
           "OpenFHE scaling technique (`fixed-manual`, `fixed-auto`, `flexible-auto`, `flexible-auto-ext`, `composite-auto`, `composite-manual`, `no-rescale`)">,
  ];
}

def ConfigureCryptoContext : Pass<"openfhe-configure-crypto-context"> {
  let summary = "Configure the crypto context in OpenFHE";
  let description = [{
    This pass generates helper functions to generate and configure the OpenFHE crypto context for the given function. Generating the crypto context sets the appropriate encryption parameters, while the configuration generates the necessary evaluation keys (relinearization and rotation keys).

    For CKKS, this pass consumes the modulus chain, scaling technique, and
    feature requirements already attached to the IR by earlier management and
    OpenFHE-specific adjustment passes. It does not re-derive CKKS structured-op
    depth requirements or choose a new modulus chain.

    For the options, reader can refer to the OpenFHE documentation at
    https://github.com/openfheorg/openfhe-development/blob/main/src/pke/examples/README.md#description-of-the-cryptocontext-parameters-and-their-restrictions

    For example, for an MLIR function `@my_func`, the generated helpers have the following signatures
    ```mlir
    func.func  @my_func__generate_crypto_context() -> !openfhe.crypto_context

    func.func  @my_func__configure_crypto_context(!openfhe.crypto_context, !openfhe.private_key) -> !openfhe.crypto_context
    ```
  }];
  let dependentDialects = ["mlir::heir::openfhe::OpenfheDialect"];
  let options = [
    Option<"entryFunction", "entry-function", "std::string",
           /*default=*/"", "Default entry function "
           "name of entry function.">,
    Option<"mulDepth", "mul-depth", "int",
           /*default=*/"0", "Manually specify the mul depth">,
    // Options for GenParamsOp
    Option<"ringDim", "ring-dim", "int",
           /*default=*/"0", "Manually specify the ring dimension (insecure is implied)">,
    // plaintext modulus is from IR
    Option<"batchSize", "batch-size", "int",
           /*default=*/"0", "Manually specify the batch size">,
    Option<"firstModSize", "first-mod-size", "int",
           /*default=*/"0", "Manually specify the first mod size">,
    Option<"scalingModSize", "scaling-mod-size", "int",
           /*default=*/"0", "Manually specify the scaling mod size">,
    Option<"digitSize", "digit-size", "int",
           /*default=*/"0", "Manually specify the digit size for relinearization">,
    Option<"numLargeDigits", "num-large-digits", "int",
           /*default=*/"0", "Manually specify the number of large digits for HYBRID relinearization">,
    Option<"maxRelinSkDeg", "max-relin-sk-deg", "int",
           /*default=*/"0", "Manually specify the max number of relin sk deg">,
    Option<"insecure", "insecure", "bool",
           /*default=*/"false", "Whether to use insecure parameter (defaults to false)">,
    Option<"keySwitchingTechniqueBV", "key-switching-technique-bv", "bool",
           /*default=*/"false", "Whether to use BV key switching technique (defaults to false)">,
    Option<"scalingTechnique", "scaling-technique", "std::string",
           /*default=*/"\"\"", "Scaling technique override (`fixed-manual`, `fixed-auto`, `flexible-auto`, `flexible-auto-ext`, `composite-auto`, `composite-manual`, `no-rescale`)">,
    // For bootstrapping
    Option<"levelBudgetEncode", "level-budget-encode", "int",
           /*default=*/"3", "Level budget for CKKS bootstrap encode (s2c) phase">,
    Option<"levelBudgetDecode", "level-budget-decode", "int",
           /*default=*/"3", "Level budget for CKKS bootstrap decode (c2s) phase">,
  ];
}

def CountAddAndKeySwitch : Pass<"openfhe-count-add-and-key-switch"> {
  let summary = "Count the number of add and key-switch operations in OpenFHE";
  let description = [{
    This pass counts the number of add and key-switch operations in the given function.

    This is used for setting the EvalAddCount and EvalKeySwitchCount in OpenFHE library.
    Cf. [Alexandru et al. 2024](https://eprint.iacr.org/2024/203) for why this
    is important for security.

    The detailed definition of these counts could be found in the KPZ21 paper
    [Revisiting Homomorphic Encryption Schemes for Finite Fields](https://ia.cr/2021/204)

    The pass should be run at the secret arithmetic level when management operations
    have been inserted and the IR is stable.
  }];
}

def FastRotationPrecompute : Pass<"openfhe-fast-rotation-precompute"> {
  let summary = "Identify and apply EvalFastRotation when possible.";
  let description = [{
    This pass identifies when a ciphertext is rotated by multiple different
    shifts, and replaces the `EvalRot` ops with `EvalFastRotationPrecompute`
    followed by `EvalFastRotate`.
  }];
}

def ConvertToExtendedBasis : Pass<"openfhe-convert-to-extended-basis"> {
  let summary = "Convert rotation operations to extended (P*Q) basis";
  let description = [{
    This pass converts `openfhe.fast_rotation` operations to use extended basis
    operations. Each `fast_rotation` is replaced with `fast_rotation_ext` followed
    by `key_switch_down`, enabling subsequent optimization to defer key-switches.
  }];
  let dependentDialects = ["mlir::heir::openfhe::OpenfheDialect"];
}

def HoistKeySwitchDown : Pass<"openfhe-hoist-key-switch-down"> {
  let summary = "Push key_switch_down operations later in the IR";
  let description = [{
    Greedily pushes `key_switch_down` past Add, AddInPlace, and Mul operations
    to reduce total key-switch count. When both operands of a binary operation
    are `key_switch_down`, this pass swaps them so the operation is performed
    in extended basis followed by a single `key_switch_down`.
  }];
  let dependentDialects = ["mlir::heir::openfhe::OpenfheDialect"];
}

def AllocToInPlace : Pass<"openfhe-alloc-to-inplace"> {
  let summary = "Utilize OpenFHE's in-place operations when possible";
  let description = [{
    This pass converts OpenFHE ops that return new ciphertexts to in-place ops.
  }];
  let dependentDialects = ["mlir::heir::openfhe::OpenfheDialect"];
}

#endif  // LIB_DIALECT_OPENFHE_TRANSFORMS_PASSES_TD_
