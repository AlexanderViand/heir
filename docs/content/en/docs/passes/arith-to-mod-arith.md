---
title: arith-to-mod-arith
weight: 52
---

## Overview

Converts standard arithmetic operations from the `arith` dialect to modular
arithmetic operations using the `mod_arith` dialect. This pass is essential for
neural network model compilation and enables efficient multiply-accumulate
operation detection and optimization.

## Input/Output

- **Input**: IR with standard arithmetic operations (`arith.addi`, `arith.muli`,
  etc.)
- **Output**: IR with modular arithmetic operations (`mod_arith.add`,
  `mod_arith.mul`, etc.) operating within a specified modulus

## Options

- `--modulus`: Modulus to use for the mod-arith dialect (default: 0, uses
  natural modulus for integer type)

## Usage Examples

```bash
# Basic usage with automatic modulus selection
heir-opt --arith-to-mod-arith input.mlir

# Specify custom modulus
heir-opt --arith-to-mod-arith=modulus=65537 input.mlir

# Neural network model compilation pipeline
heir-opt --arith-to-mod-arith --find-mac --arith-to-cggi-quart input.mlir

# Full TOSA to CGGI pipeline
heir-opt --tosa-to-arith --arith-to-mod-arith --arith-to-cggi input.mlir
```

## When to Use

- **Neural Network Compilation**: Essential for TOSA neural network model
  lowering to CGGI backends
- **MAC Optimization**: Required before multiply-accumulate operation detection
  with `find-mac`
- **Modular Arithmetic**: When operations need to be performed within a specific
  modulus
- **Precision Management**: For controlling arithmetic precision and overflow
  behavior

## Prerequisites

- Input should contain standard arithmetic operations from the `arith` dialect
- Often used as the first step in neural network compilation pipelines
- Compatible with TOSA dialect operations after `tosa-to-arith`

## Technical Notes

- **Modulus Selection**: When modulus is not specified (default 0), uses the
  natural modulus for the integer type
- **Neural Network Focus**: Specifically designed for neural network model
  compilation workflows
- **MAC Preparation**: Transforms operations to enable multiply-accumulate
  pattern detection
- **Precision Control**: Large precision operations (64-bit, 32-bit) can be
  later lowered to smaller precision (8-bit, 4-bit)

## Neural Network Compilation

- **TOSA Integration**: Works with TOSA (Tensor Operator Set Architecture)
  neural network models
- **MAC Detection**: Enables the `find-mac` pass to detect multiply-accumulate
  patterns
- **Precision Lowering**: Prepares operations for subsequent precision reduction
  passes
- **CGGI Backend**: Part of the pipeline for targeting CGGI FHE backends

## Modulus Behavior

- **Automatic Selection**: Default behavior uses natural modulus for each
  integer type
- **Custom Modulus**: Can specify custom modulus for specific arithmetic
  requirements
- **Overflow Handling**: Modular arithmetic provides well-defined overflow
  behavior

## Related Passes

- `tosa-to-arith`: Converts TOSA operations to arithmetic (may run before)
- `find-mac`: Detects multiply-accumulate patterns (should run after)
- `arith-to-cggi-quart`: Precision decomposition for CGGI targeting (may run
  after)
- `mod-arith-to-arith`: Reverse conversion back to standard arithmetic
