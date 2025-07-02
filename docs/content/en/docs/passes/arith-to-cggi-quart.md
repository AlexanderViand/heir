---
title: arith-to-cggi-quart
weight: 51
---

## Overview

Converts high-precision arithmetic operations (32-bit) to CGGI boolean
operations by dividing each operation into smaller parts (8-bit components).
This pass enables more efficient boolean circuit evaluation by working with
smaller precision arithmetic operations that map better to CGGI operations.

## Input/Output

- **Input**: IR with high-precision arithmetic operations (typically 32-bit
  `arith` operations)
- **Output**: IR with CGGI boolean operations implementing 8-bit arithmetic
  components, using tensors to represent the decomposed values

## Options

This pass does not currently expose command-line options.

## Usage Examples

```bash
# Basic usage for precision decomposition
heir-opt --arith-to-cggi-quart input.mlir

# Pipeline for neural network models
heir-opt --arith-to-mod-arith --arith-to-cggi-quart input.mlir

# Full pipeline with MAC optimization
heir-opt --arith-to-mod-arith --find-mac --arith-to-cggi-quart input.mlir
```

## When to Use

- **Neural Network Inference**: When lowering TOSA neural network models to CGGI
  backends
- **High-Precision Arithmetic**: When working with 32-bit operations that need
  boolean circuit implementation
- **Memory Efficiency**: When smaller precision operations are more efficient
  for the target FHE backend
- **MAC Operations**: When combined with multiply-accumulate optimization

## Prerequisites

- Input should contain high-precision arithmetic operations (typically 32-bit)
- Works well after `arith-to-mod-arith` for modular arithmetic conversion
- May benefit from `find-mac` pass to optimize multiply-accumulate patterns

## Technical Notes

- **Precision Decomposition**: Splits 32-bit integers into four 8-bit components
- **Tensor Representation**: Uses tensor dialect to store decomposed values
- **Carry Handling**: Stores components in 16-bit integers to preserve carry
  information
- **LSB First**: First tensor element corresponds to least significant bits
- **Based on MLIR**: Inspired by the `arith-emulate-wide-int` pass from MLIR

## Implementation Details

- **32-bit → 4×8-bit**: Decomposes 32-bit operations into 8-bit components
- **16-bit Storage**: Each 8-bit component stored in 16-bit integer for carry
  bits
- **Tensor Operations**: Uses tensor dialect for managing decomposed arithmetic
- **CGGI Targeting**: Final output targets CGGI boolean operations

## Neural Network Support

- **TOSA Lowering**: Essential for neural network model compilation
- **MAC Optimization**: Works with multiply-accumulate operation detection
- **Efficient Inference**: Enables FHE-based neural network inference

## Related Passes

- `arith-to-mod-arith`: Modular arithmetic conversion (should run before)
- `find-mac`: Multiply-accumulate optimization (may run before)
- `arith-to-cggi`: Alternative direct arithmetic to CGGI conversion
- `yosys-optimizer`: Boolean circuit optimization (may run after)
