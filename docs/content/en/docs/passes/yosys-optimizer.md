---
title: Yosys Optimizer
weight: 20
---

## Overview

The `yosys-optimizer` pass integrates the external Yosys EDA tool to convert
arithmetic circuits to optimized boolean circuits. This transformation is
crucial for boolean FHE schemes like CGGI and TFHE, where operations are
performed at the bit level rather than on packed integers.

## Input/Output

- **Input**: MLIR IR containing `secret.generic` operations with arithmetic
  operations
- **Output**: MLIR IR with optimized boolean circuits using arith and comb
  dialects, where multi-bit integers are converted to `tensor<N xi1>`

## Options

- **`abc-fast`**: Run ABC optimizer in fast mode for quicker compilation at the
  cost of potentially larger circuits
- **`unroll-factor`**: Unroll loops by the specified factor before optimization
  (optional)
- **`print-stats`**: Print circuit size statistics after optimization
- **`mode`**: Choose between `Boolean` (boolean gates) or `LUT` (lookup table
  gates) mapping
- **`use-submodules`**: Extract generic operation bodies into submodules for
  better organization

## Usage Examples

```bash
# Basic optimization with default settings
heir-opt --yosys-optimizer input.mlir

# Fast optimization mode
heir-opt --yosys-optimizer=abc-fast input.mlir

# With loop unrolling and statistics
heir-opt --yosys-optimizer=unroll-factor=4,print-stats input.mlir

# LUT-based optimization with submodules
heir-opt --yosys-optimizer=mode=LUT,use-submodules input.mlir
```

## When to Use

This pass is essential for:

1. **Boolean FHE schemes**: When targeting CGGI, TFHE, or other bit-level FHE
   implementations
1. **Circuit optimization**: To minimize boolean circuit size and depth for
   performance
1. **Hardware synthesis**: When generating circuits for FPGA or ASIC
   implementations
1. **Bit-level operations**: When algorithms naturally operate on individual
   bits
1. **EDA tool integration**: Leveraging advanced optimization algorithms from
   the EDA community

**Prerequisites**: Requires Yosys to be installed and available in the system
PATH.

## Implementation Details

**Transformation Process:**

1. **Circuit Extraction**: Identifies `secret.generic` operations containing
   arithmetic operations
1. **Booleanization**: Converts multi-bit arithmetic to bit-level operations
   (e.g., `i8` → `tensor<8xi1>`)
1. **Yosys Integration**: Exports circuits to Yosys format for optimization
1. **Optimization**: Applies Yosys optimization algorithms (ABC, etc.)
1. **Import**: Converts optimized circuits back to MLIR representation
1. **Integration**: Replaces original operations with optimized boolean circuits

**Type Transformations:**

```mlir
// Input: Multi-bit arithmetic
%result = arith.addi %a, %b : i32

// Output: Bit-level operations
%result = comb.add %a, %b : tensor<32xi1>
```

**Optimization Modes:**

**ABC Fast Mode:**

- Trades optimization quality for compilation speed
- Suitable for rapid prototyping and development
- May result in larger circuits but compiles much faster

**Standard Mode:**

- Applies comprehensive optimization algorithms
- Produces smaller, more efficient circuits
- Takes longer compilation time

**LUT vs Boolean Mode:**

- **Boolean**: Maps to individual boolean gates (AND, OR, XOR, etc.)
- **LUT**: Maps to lookup table implementations, better for FPGA targets

**Submodule Extraction:**

- Useful for large circuits with isolated generic operations
- Improves circuit organization and potential reuse
- Should not be used with distributed generics through loops

**Performance Considerations:**

- Circuit size statistics help evaluate optimization effectiveness
- Loop unrolling can expose more optimization opportunities
- Trade-offs between compilation time, circuit size, and circuit depth

**Integration Requirements:**

- External dependency on Yosys EDA tool
- Proper PATH configuration for tool invocation
- Compatible with MLIR's external tool integration framework
