---
title: secret-to-cggi
weight: 12
---

## Overview

Converts `secret.generic` operations containing boolean operations to CGGI
(Chillotti-Gama-Georgieva-Izabachène) operations for efficient boolean circuit
evaluation over encrypted data. This pass is optimized for applications
requiring bitwise operations and boolean logic on encrypted data.

## Input/Output

- **Input**: IR with `secret.generic` blocks containing boolean operations on
  secret values, often from combinational logic operations
- **Output**: IR with CGGI dialect operations (`cggi.lut3`, `cggi.and`, etc.)
  for encrypted boolean circuit evaluation

## Options

This pass does not currently expose command-line options.

## Usage Examples

```bash
# Basic usage
heir-opt --secret-to-cggi input.mlir

# Common pipeline: optimize boolean circuits first, then convert
heir-opt --yosys-optimizer --secret-distribute-generic --canonicalize --secret-to-cggi input.mlir

# Pipeline for boolean circuit evaluation
heir-opt --comb-to-cggi --secret-to-cggi input.mlir
```

## When to Use

- When working with boolean circuits that need to operate on encrypted data
- For bitwise operations and combinational logic on secret values
- When targeting TFHE-based backends that excel at boolean operations
- After boolean circuit optimization with Yosys or similar tools
- For applications requiring encrypted bit manipulation

## Prerequisites

- Input should contain boolean or combinational logic operations
- Works best with circuits optimized using the `yosys-optimizer` pass
- Should run `secret-distribute-generic` and `canonicalize` beforehand
- Compatible with outputs from the `comb` dialect

## Technical Notes

- **LUT3 Operations**: Uses 3-input lookup tables for efficient evaluation of
  arbitrary 3-input boolean functions
- **Noise Management**: Optimized for boolean operations with minimal noise
  growth
- **Circuit Optimization**: Works best with circuits that have been optimized
  for boolean gate density
- **Memory Operations**: Supports memref operations for bit-level data
  manipulation

## Related Passes

- `yosys-optimizer`: Boolean circuit optimization (should run before)
- `secret-distribute-generic`: Distributes secret operations (should run before)
- `cggi-to-tfhe-rust`: Lowers CGGI to TfheRust backend (typically runs after)
- `cggi-to-jaxite`: Alternative backend targeting (typically runs after)
- `comb-to-cggi`: Converts combinational logic to CGGI operations
