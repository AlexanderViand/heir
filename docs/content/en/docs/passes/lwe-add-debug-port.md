---
title: LWE Add Debug Port
weight: 11
---

## Overview

The `lwe-add-debug-port` pass adds debug ports to LWE (Learning With Errors)
encrypted functions to enable runtime debugging and analysis of homomorphic
computations. This pass inserts debug calls after each homomorphic operation,
allowing developers to inspect ciphertext states during execution.

## Input/Output

- **Input**: MLIR module with LWE dialect operations in functions
- **Output**: Same module with debug port function declarations and debug calls
  inserted after each homomorphic operation

## Options

- `--entry-function` (string, default: ""): Name of the entry function to
  instrument with debug ports
  - If empty, all functions containing LWE operations will be instrumented

## Usage Examples

```bash
# Add debug ports to all functions with LWE operations
heir-opt --lwe-add-debug-port input.mlir

# Add debug ports only to a specific function
heir-opt --lwe-add-debug-port --entry-function=main input.mlir

# Add debug ports to a specific computation function
heir-opt --lwe-add-debug-port --entry-function=encrypted_computation input.mlir
```

## When to Use

This pass should be used during development and debugging phases:

1. When debugging homomorphic encryption implementations
1. For analyzing ciphertext evolution through computations
1. When validating encryption/decryption correctness
1. For performance profiling of homomorphic operations

**Note**: This pass is intended for development and testing purposes only. Debug
ports should be removed before production deployment.

## Debug Function Requirements

Users must provide implementations for the debug functions declared by this
pass. The debug functions follow the naming convention `__heir_debug` and must
be implemented in the target application.

## Implementation Details

The pass performs the following transformations:

1. **Add Secret Key Parameter**: Modifies function signatures to include secret
   key parameters
1. **Declare Debug Function**: Adds external debug function declarations
1. **Insert Debug Calls**: Places debug calls after each homomorphic operation
1. **Preserve Operation Results**: Ensures debug calls don't interfere with
   computation flow

## Example Transformation

**Input:**

```mlir
func.func @encrypted_add(%arg0: !lwe.lwe_ciphertext, %arg1: !lwe.lwe_ciphertext) -> !lwe.lwe_ciphertext {
  %0 = lwe.radd %arg0, %arg1 : !lwe.lwe_ciphertext
  %1 = lwe.rmul %0, %arg1 : !lwe.lwe_ciphertext
  return %1 : !lwe.lwe_ciphertext
}
```

**Output:**

```mlir
// Declaration of external debug function
func.func private @__heir_debug(!lwe.lwe_secret_key, !lwe.lwe_ciphertext)

// Modified function with secret key parameter and debug calls
func.func @encrypted_add(%sk: !lwe.lwe_secret_key, %arg0: !lwe.lwe_ciphertext, %arg1: !lwe.lwe_ciphertext) -> !lwe.lwe_ciphertext {
  %0 = lwe.radd %arg0, %arg1 : !lwe.lwe_ciphertext
  // Debug call after addition
  func.call @__heir_debug(%sk, %0) : (!lwe.lwe_secret_key, !lwe.lwe_ciphertext) -> ()

  %1 = lwe.rmul %0, %arg1 : !lwe.lwe_ciphertext
  // Debug call after multiplication
  func.call @__heir_debug(%sk, %1) : (!lwe.lwe_secret_key, !lwe.lwe_ciphertext) -> ()

  return %1 : !lwe.lwe_ciphertext
}
```

## Debug Function Implementation

Users should implement the debug function in their application code. Example
implementation:

```cpp
// C++ implementation example
void __heir_debug(const LweSecretKey& sk, const LweCiphertext& ct) {
    // Decrypt ciphertext for debugging
    auto plaintext = decrypt(sk, ct);
    std::cout << "Debug: Decrypted value = " << plaintext << std::endl;

    // Additional debugging logic:
    // - Noise level analysis
    // - Ciphertext validity checks
    // - Performance measurements
}
```

## Debugging Capabilities

With debug ports, developers can:

- **Decrypt and Inspect**: View intermediate plaintext values
- **Noise Analysis**: Monitor noise growth throughout computation
- **Correctness Validation**: Verify intermediate results match expectations
- **Performance Profiling**: Measure operation execution times
- **Error Detection**: Identify where computations go wrong
