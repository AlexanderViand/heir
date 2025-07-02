---
title: Add Client Interface
weight: 19
---

## Overview

The `add-client-interface` pass generates encrypt and decrypt functions for
compiled FHE functions, maintaining the original function interface while
handling the complex data layout transformations required by ciphertext packing.
This is essential for client-server FHE applications.

## Input/Output

- **Input**: MLIR IR containing compiled secret functions with transformed
  signatures
- **Output**: MLIR IR with additional encrypt/decrypt functions that preserve
  the original interface

## Options

- **`ciphertext-size`**: Power of two length of the ciphertexts for data packing
  (default: 1024)

## Usage Examples

```bash
# Add client interface with default ciphertext size
heir-opt --add-client-interface input.mlir

# Specify custom ciphertext size
heir-opt --add-client-interface=ciphertext-size=2048 input.mlir

# Common usage in FHE compilation pipeline
heir-opt --convert-to-ciphertext-semantics --add-client-interface input.mlir
```

## When to Use

This pass is essential for:

1. **Client-server FHE applications**: When creating separate client and server
   components
1. **Interface preservation**: To maintain usable APIs after complex ciphertext
   transformations
1. **Data layout management**: When scalar values are packed into ciphertext
   tensors
1. **Cross-backend compatibility**: For supporting both encryption and plaintext
   testing backends
1. **Production FHE deployment**: When deploying FHE applications that need
   clean client interfaces

The pass should be used late in the compilation pipeline, after most
optimizations but before final backend lowering.

## Implementation Details

The transformation process:

1. **Function Analysis**: Identifies compiled functions with transformed
   signatures due to ciphertext packing
1. **Interface Preservation**: Analyzes original function interfaces from
   metadata or type annotations
1. **Encode Function Generation**: Creates functions that convert client data to
   packed ciphertext format
1. **Decode Function Generation**: Creates functions that convert packed results
   back to client format
1. **Layout Management**: Handles the conversion between scalar and tensor
   representations

**Generated Functions:**

- **Encrypt/Encode**: Converts client inputs to the ciphertext-packed format
  expected by the compiled function
- **Decrypt/Decode**: Converts the compiled function's output back to the
  original return type format

**Key Features:**

- **Type Preservation**: Maintains original scalar types in client interface
  while handling tensor packing internally
- **Backend Agnostic**: Works with both actual encryption backends and plaintext
  testing backends
- **Layout Abstraction**: Hides complex ciphertext packing details from client
  code
- **Multiple Functions**: Generates interfaces for all compiled functions in the
  module

**Example Transformation:**

```mlir
// Original function (transformed by earlier passes):
func.func @compute_packed(%arg0: !secret.secret<tensor<1024xi32>>) -> !secret.secret<tensor<1024xi32>>

// Generated client interface:
func.func @compute_encrypt(%arg0: i32) -> !secret.secret<tensor<1024xi32>> {
  // Encoding logic: scalar -> packed tensor
}

func.func @compute_decrypt(%arg0: !secret.secret<tensor<1024xi32>>) -> i32 {
  // Decoding logic: packed tensor -> scalar
}
```

**Integration Considerations:**

- Must understand the packing scheme used by earlier passes
- Coordinates with backend-specific encoding/decoding requirements
- Preserves security properties while providing usable interfaces
