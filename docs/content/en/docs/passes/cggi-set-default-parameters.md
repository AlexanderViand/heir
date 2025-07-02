---
title: CGGI Set Default Parameters
weight: 4
---

## Overview

The `cggi-set-default-parameters` pass adds default parameters to all CGGI
(CGGI/TFHE) dialect operations as `cggi_params` named attributes. This pass
provides a quick way to attach working parameters to CGGI operations for testing
and development purposes.

## Input/Output

- **Input**: MLIR module with CGGI dialect operations (e.g., `cggi.and`,
  `cggi.xor`, `cggi.lut2`, etc.)
- **Output**: Same module with `cggi_params` attributes added to all CGGI
  operations, overriding any existing parameters

## Options

This pass has no configurable options. The parameters are hard-coded in the
implementation.

## Usage Examples

```bash
# Add default parameters to all CGGI operations
heir-opt --cggi-set-default-parameters input.mlir
```

## When to Use

This pass should be used:

- **For testing and prototyping**: When you need to quickly add working
  parameters to CGGI operations
- **As a parameter provider**: Before proper parameter selection mechanisms are
  implemented
- **During development**: When experimenting with CGGI transformations and need
  valid parameters

**Important**: This pass is primarily for testing purposes and should **NOT** be
used in production. The hard-coded parameters may not be optimal or secure for
real-world applications.

## Prerequisites

- Input module must contain CGGI dialect operations
- No specific pass ordering requirements

## Parameter Details

The specific parameters are defined in the implementation file
`lib/Dialect/CGGI/Transforms/SetDefaultParameters.cpp`. These parameters
include:

- LWE dimension and noise parameters
- Bootstrapping key parameters
- Modulus and other cryptographic parameters

## Example Transformation

**Input:**

```mlir
module {
  func.func @example(%arg0: !lwe.lwe_ciphertext, %arg1: !lwe.lwe_ciphertext) -> !lwe.lwe_ciphertext {
    %0 = cggi.and %arg0, %arg1 : !lwe.lwe_ciphertext
    return %0 : !lwe.lwe_ciphertext
  }
}
```

**Output:**

```mlir
module {
  func.func @example(%arg0: !lwe.lwe_ciphertext, %arg1: !lwe.lwe_ciphertext) -> !lwe.lwe_ciphertext {
    %0 = cggi.and %arg0, %arg1 {cggi_params = #cggi.cggi_params<...>} : !lwe.lwe_ciphertext
    return %0 : !lwe.lwe_ciphertext
  }
}
```

## Relationship to Other Passes

- Use before `--cggi-decompose-operations` to ensure all operations have
  parameters before decomposition
- Alternative to proper parameter generation passes (which should be used in
  production)
- Overrides any existing `cggi_params` attributes

## Development Notes

- Parameters are hard-coded for simplicity and should be updated as CGGI
  parameter research advances
- This pass will likely be deprecated once automatic parameter generation for
  CGGI is implemented
- Consider using proper parameter selection tools for production workloads
