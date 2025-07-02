---
title: lwe-to-openfhe
weight: 30
---

## Overview

Converts generic LWE dialect operations to OpenFHE library-specific operations,
enabling code generation for one of the most widely used FHE implementations.
This pass generates efficient C++ code that integrates with OpenFHE's
optimization features and supports both exact (BGV) and approximate (CKKS)
arithmetic modes.

## Input/Output

- **Input**: IR with generic LWE dialect operations (`lwe.add`, `lwe.mul`, etc.)
- **Output**: IR with OpenFHE dialect operations (`openfhe.add`, `openfhe.mul`,
  etc.) that generate OpenFHE C++ API calls

## Options

This pass does not currently expose command-line options.

## Usage Examples

```bash
# Basic usage for OpenFHE backend targeting
heir-opt --lwe-to-openfhe input.mlir

# Complete pipeline from secret to OpenFHE (BGV)
heir-opt --secret-to-bgv --bgv-to-lwe --lwe-to-openfhe input.mlir

# Complete pipeline from secret to OpenFHE (CKKS)
heir-opt --secret-to-ckks --ckks-to-lwe --lwe-to-openfhe input.mlir

# Full pipeline with optimizations
heir-opt --secret-to-bgv --bgv-to-lwe --canonicalize --lwe-to-openfhe input.mlir
```

## When to Use

- **OpenFHE Backend**: When targeting OpenFHE as the FHE execution library
- **Production Deployment**: For generating production-ready FHE code with
  OpenFHE optimizations
- **Performance Requirements**: When leveraging OpenFHE's built-in performance
  optimizations
- **Industry Standard**: When using a well-established, battle-tested FHE
  library

## Prerequisites

- Input should contain generic LWE dialect operations
- Should run after scheme-to-LWE conversion passes (`bgv-to-lwe`, `ckks-to-lwe`)
- OpenFHE library must be available for linking the generated code

## Technical Notes

- **Code Generation**: Produces efficient OpenFHE C++ API calls with proper
  memory management
- **Dual Mode Support**: Handles both BGV (exact) and CKKS (approximate)
  arithmetic modes
- **Context Management**: Automatically handles OpenFHE context creation and
  parameter setup
- **SIMD Support**: Compatible with tensor operations for SIMD-packed ciphertext
  operations
- **Optimization Integration**: Leverages OpenFHE's built-in optimizations and
  parameter tuning

## Current Limitations

- Temporarily includes patterns for direct CKKS/BGV operations while LWE
  abstraction is completed (TODO #1193)
- Generated code requires linking against OpenFHE library
- Some advanced OpenFHE features may not be exposed through the pass

## Performance Features

- **Threading Support**: Compatible with OpenFHE's parallel execution
- **Memory Efficiency**: Uses OpenFHE's optimized memory management patterns
- **Hardware Acceleration**: Supports OpenFHE's hardware acceleration features
- **Parameter Optimization**: Integrates with OpenFHE's automatic parameter
  tuning

## Related Passes

- `bgv-to-lwe`: Prepares BGV operations for OpenFHE targeting (should run
  before)
- `ckks-to-lwe`: Prepares CKKS operations for OpenFHE targeting (should run
  before)
- `lwe-to-lattigo`: Alternative backend targeting for Go-based applications
- `generate-param-bgv`: Parameter generation for BGV schemes (may run before)
- `generate-param-ckks`: Parameter generation for CKKS schemes (may run before)
