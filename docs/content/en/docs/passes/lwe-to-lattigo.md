---
title: lwe-to-lattigo
weight: 31
---

## Overview

Converts generic LWE dialect operations to Lattigo library-specific operations,
enabling code generation for the high-performance Go-based FHE implementation.
This pass generates idiomatic Go code that leverages Lattigo's optimizations and
integrates well with Go's concurrency features and ecosystem.

## Input/Output

- **Input**: IR with generic LWE dialect operations (`lwe.add`, `lwe.mul`, etc.)
- **Output**: IR with Lattigo dialect operations (`lattigo.add`, `lattigo.mul`,
  etc.) that generate Lattigo Go API calls

## Options

This pass does not currently expose command-line options.

## Usage Examples

```bash
# Basic usage for Lattigo backend targeting
heir-opt --lwe-to-lattigo input.mlir

# Complete pipeline from secret to Lattigo (BGV)
heir-opt --secret-to-bgv --bgv-to-lwe --lwe-to-lattigo input.mlir

# Complete pipeline from secret to Lattigo (CKKS)
heir-opt --secret-to-ckks --ckks-to-lwe --lwe-to-lattigo input.mlir

# Full pipeline with optimizations
heir-opt --secret-to-bgv --bgv-to-lwe --canonicalize --lwe-to-lattigo input.mlir
```

## When to Use

- **Go Integration**: When targeting Go-based applications or services
- **Lattigo Backend**: When using Lattigo as the FHE execution library
- **Concurrency**: When leveraging Go's goroutines for parallel FHE computation
- **Ecosystem Integration**: When integrating with Go's rich standard library
  and ecosystem
- **Clean Codebase**: When prioritizing readable, maintainable generated code

## Prerequisites

- Input should contain generic LWE dialect operations
- Should run after scheme-to-LWE conversion passes (`bgv-to-lwe`, `ckks-to-lwe`)
- Lattigo library must be available as a Go module dependency
- Go build toolchain required for compilation

## Technical Notes

- **Go Code Generation**: Produces idiomatic Lattigo Go API calls with proper
  error handling
- **Dual Scheme Support**: Handles both BGV (exact) and CKKS (approximate)
  arithmetic modes
- **Context Management**: Automatically handles Lattigo context creation and
  parameter setup
- **Concurrency Support**: Compatible with Go's goroutines for parallel
  computation
- **Memory Management**: Leverages Go's garbage collection for automatic memory
  management

## Go Language Benefits

- **Garbage Collection**: Automatic memory management eliminates manual memory
  handling
- **Concurrency**: Native support for parallel FHE operations through goroutines
- **Error Handling**: Clean, explicit error handling patterns
- **Standard Library**: Rich ecosystem of Go packages and tools
- **Build System**: Seamless integration with Go modules and build toolchain

## Performance Features

- **High Performance**: Leverages Lattigo's optimized Go implementations
- **Parallel Execution**: Supports concurrent operations through goroutines
- **Vectorization**: Compatible with Lattigo's SIMD optimization features
- **Efficient Memory Usage**: Optimized patterns for Go runtime performance

## Related Passes

- `bgv-to-lwe`: Prepares BGV operations for Lattigo targeting (should run
  before)
- `ckks-to-lwe`: Prepares CKKS operations for Lattigo targeting (should run
  before)
- `lwe-to-openfhe`: Alternative backend targeting for C++ applications
- `generate-param-bgv`: Parameter generation for BGV schemes (may run before)
- `generate-param-ckks`: Parameter generation for CKKS schemes (may run before)
