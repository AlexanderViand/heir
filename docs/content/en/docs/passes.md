---
title: Passes
weight: 70
---

HEIR provides a comprehensive set of transformation passes for compiling
high-level computations to various FHE backends. This page provides an overview
of the pass system and links to detailed documentation for specific passes.

## Pass Categories

### Conversion Passes

Convert between different dialect representations in the compilation pipeline.

- [secret-to-bgv](passes/secret-to-bgv/): Convert secret operations to BGV
  ciphertext operations
- [secret-to-ckks](passes/secret-to-ckks/): Convert secret operations to CKKS
  ciphertext operations
- [secret-to-cggi](passes/secret-to-cggi/): Convert secret operations to CGGI
  boolean operations
- [secret-to-mod-arith](passes/secret-to-mod-arith/): Convert secret operations
  to plaintext modular arithmetic
- [bgv-to-lwe](passes/bgv-to-lwe/): Convert BGV operations to LWE dialect

### Optimization Passes

Optimize the intermediate representation for better performance.

### Analysis Passes

Analyze properties of the computation for optimization and validation.

## Using Passes

Passes are invoked using the `heir-opt` tool:

```bash
heir-opt --pass-name input.mlir
```

Multiple passes can be chained together:

```bash
heir-opt --pass1 --pass2 --pass3 input.mlir
```

## Common Pass Sequences

### FHE Compilation Pipeline

```bash
heir-opt --secret-distribute-generic --canonicalize --secret-to-bgv --bgv-to-lwe --lwe-to-openfhe input.mlir
```

### Boolean Circuit Pipeline

```bash
heir-opt --yosys-optimizer --secret-distribute-generic --secret-to-cggi --cggi-to-tfhe-rust input.mlir
```

### Plaintext Testing Pipeline

```bash
heir-opt --secret-distribute-generic --secret-to-mod-arith=modulus=17 input.mlir
```
