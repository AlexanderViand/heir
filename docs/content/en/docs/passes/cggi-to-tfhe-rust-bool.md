---
title: cggi-to-tfhe-rust-bool
weight: 41
---

## Overview

Converts CGGI dialect operations to TFHE-rs boolean-specific operations,
providing a specialized backend targeting for pure boolean operations with
optimized performance and minimal memory overhead. This pass is specifically
designed for boolean-only computations, offering better performance than general
TFHE-rs targeting.

## Input/Output

- **Input**: IR with CGGI dialect operations (`cggi.lut3`, `cggi.and`, etc.)
  that represent pure boolean circuits
- **Output**: IR with TFHE-rs boolean dialect operations optimized for
  boolean-only computation patterns

## Options

This pass does not currently expose command-line options.

## Usage Examples

```bash
# Basic usage for boolean-optimized TFHE-rs targeting
heir-opt --cggi-to-tfhe-rust-bool input.mlir

# Pipeline for pure boolean circuits
heir-opt --secret-to-cggi --cggi-to-tfhe-rust-bool input.mlir

# Optimized boolean circuit pipeline
heir-opt --yosys-optimizer --secret-to-cggi --cggi-to-tfhe-rust-bool input.mlir
```

## When to Use

- **Pure Boolean Circuits**: When working exclusively with boolean operations
  without arithmetic mixing
- **Performance Critical**: When requiring maximum performance for boolean
  operations
- **Memory Constrained**: When minimizing memory usage is important
- **Boolean-Only Workloads**: For applications that don't require mixed
  arithmetic and boolean operations

## Prerequisites

- Input should contain CGGI dialect operations representing pure boolean
  circuits
- Should run after `secret-to-cggi` or similar CGGI generation passes
- TFHE-rs crate with boolean specializations must be available
- Rust toolchain required for compilation

## Technical Notes

- **Boolean Specialization**: Optimized specifically for pure boolean operations
  without arithmetic mixing
- **Memory Optimization**: Specialized memory layout for boolean values with
  minimal overhead
- **Performance Focus**: Streamlined API calls for boolean-only computation
  patterns
- **Reduced Overhead**: Lower API call overhead compared to general TFHE-rs
  targeting

## Performance Benefits

- **Minimal Memory Footprint**: Optimized memory usage for boolean values
- **Streamlined Operations**: Reduced overhead for boolean-specific operations
- **Specialized Parameters**: Parameter handling optimized for boolean-only
  workflows
- **Better Throughput**: Higher performance than general TFHE-rs for pure
  boolean circuits

## When NOT to Use

- **Mixed Arithmetic**: When circuits contain both boolean and arithmetic
  operations
- **General Purpose**: When flexibility for multiple operation types is needed
- **Complex Data Types**: When working with non-boolean encrypted data types

## Related Passes

- `secret-to-cggi`: Generates CGGI operations (should run before)
- `cggi-to-tfhe-rust`: General TFHE-rs targeting for mixed operations
- `yosys-optimizer`: Boolean circuit optimization (may run before)
- `cggi-to-jaxite`: Alternative backend targeting for Python applications
