---
title: cggi-to-tfhe-rust
weight: 40
---

## Overview

Converts CGGI dialect operations to TFHE-rs library-specific operations,
enabling code generation for the high-performance Rust-based FHE implementation
optimized for boolean circuit evaluation. This pass generates safe, performant
Rust code that leverages TFHE-rs's optimizations for boolean operations over
encrypted data.

## Input/Output

- **Input**: IR with CGGI dialect operations (`cggi.lut3`, `cggi.and`, etc.)
- **Output**: IR with TFHE-rs dialect operations
  (`tfhe_rust.apply_lookup_table`, `tfhe_rust.scalar_and_assign`, etc.) that
  generate TFHE-rs Rust API calls

## Options

This pass does not currently expose command-line options.

## Usage Examples

```bash
# Basic usage for TFHE-rs backend targeting
heir-opt --cggi-to-tfhe-rust input.mlir

# Complete pipeline from secret to TFHE-rs
heir-opt --secret-to-cggi --cggi-to-tfhe-rust input.mlir

# Pipeline with boolean circuit optimization
heir-opt --yosys-optimizer --secret-to-cggi --cggi-to-tfhe-rust input.mlir

# Full boolean circuit pipeline
heir-opt --comb-to-cggi --cggi-to-tfhe-rust input.mlir
```

## When to Use

- **Boolean Circuits**: When implementing boolean logic operations on encrypted
  data
- **TFHE-rs Backend**: When targeting TFHE-rs as the FHE execution library
- **Rust Integration**: When building Rust-based applications with FHE
- **High Performance**: When requiring optimized boolean gate evaluation
- **Memory Safety**: When prioritizing Rust's memory safety guarantees

## Prerequisites

- Input should contain CGGI dialect operations
- Should run after `secret-to-cggi` or similar CGGI generation passes
- TFHE-rs crate must be available as a Rust dependency
- Rust toolchain required for compilation

## Technical Notes

- **Rust Code Generation**: Produces safe TFHE-rs Rust API calls with memory
  safety guarantees
- **Boolean Specialization**: Optimized specifically for boolean gate evaluation
  over encrypted data
- **LUT Support**: Efficiently handles lookup table operations for arbitrary
  boolean functions
- **Context Management**: Automatically handles TFHE-rs context creation and
  parameter setup
- **Memory Safety**: Leverages Rust's ownership model for safe resource
  management

## Boolean Circuit Features

- **LUT3 Operations**: Supports efficient 3-input lookup tables for arbitrary
  boolean functions
- **Gate Evaluation**: Optimized for boolean gate operations (AND, OR, XOR,
  etc.)
- **Combinational Logic**: Handles complex combinational logic circuits
- **Yosys Integration**: Compatible with Yosys-optimized boolean circuits

## Rust Language Benefits

- **Memory Safety**: Zero-cost memory safety guarantees without garbage
  collection
- **Performance**: Rust's zero-cost abstractions provide efficient execution
- **Ownership Model**: Safe resource management through Rust's ownership system
- **Ecosystem**: Integration with Cargo and the Rust package ecosystem
- **Concurrency**: Safe concurrent execution patterns

## Performance Features

- **High-Performance**: Leverages TFHE-rs's highly optimized boolean operations
- **Efficient Memory**: Rust's efficient memory management patterns
- **Parallel Computation**: Compatible with parallel boolean circuit evaluation
- **Zero-Cost Abstractions**: Rust's compile-time optimizations

## Related Passes

- `secret-to-cggi`: Generates CGGI operations (should run before)
- `yosys-optimizer`: Boolean circuit optimization (may run before)
- `cggi-to-jaxite`: Alternative backend targeting for Python applications
- `cggi-to-tfhe-rust-bool`: Alternative TFHE-rs targeting for boolean-specific
  operations
