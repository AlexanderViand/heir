---
title: arith-to-cggi
weight: 50
---

## Overview

Converts standard arithmetic operations from the `arith` dialect to CGGI boolean
operations, enabling boolean circuit evaluation of arithmetic computations on
encrypted data. This pass is essential for implementing arithmetic algorithms
using boolean gate operations in FHE.

## Input/Output

- **Input**: IR with standard arithmetic operations (`arith.addi`, `arith.muli`,
  etc.)
- **Output**: IR with CGGI boolean operations (`cggi.lut3`, `cggi.and`, etc.)
  that implement arithmetic through boolean circuits

## Options

This pass does not currently expose command-line options.

## Usage Examples

```bash
# Basic usage
heir-opt --arith-to-cggi input.mlir

# Pipeline for arithmetic to boolean conversion
heir-opt --arith-to-mod-arith --arith-to-cggi input.mlir

# Full pipeline with optimization
heir-opt --arith-to-mod-arith --arith-to-cggi --yosys-optimizer input.mlir
```

## When to Use

- **Arithmetic on Boolean Circuits**: When implementing arithmetic operations
  using boolean gate evaluation
- **CGGI Backend Targeting**: When using CGGI-based FHE libraries for arithmetic
  computations
- **Boolean Circuit Compilation**: For compiling arithmetic algorithms to
  boolean circuits
- **Low-Level Arithmetic**: When working with bitwise arithmetic implementations

## Prerequisites

- Input should contain standard arithmetic operations from the `arith` dialect
- May benefit from running `arith-to-mod-arith` first for modular arithmetic
- Works best with operations that can be efficiently implemented as boolean
  circuits

## Technical Notes

- **Boolean Implementation**: Converts arithmetic operations to their boolean
  circuit equivalents
- **Gate-Level Operations**: Uses boolean gates (AND, OR, XOR, etc.) to
  implement arithmetic
- **Circuit Complexity**: Arithmetic operations result in complex boolean
  circuits
- **Optimization Opportunity**: Generated circuits can benefit from boolean
  optimization passes

## Related Passes

- `arith-to-mod-arith`: May run before for modular arithmetic conversion
- `arith-to-cggi-quart`: Alternative conversion that breaks operations into
  smaller parts
- `yosys-optimizer`: Boolean circuit optimization (should run after)
- `cggi-to-tfhe-rust`: Backend targeting for CGGI operations (runs after)
