---
title: Optimize Relinearization
weight: 17
---

## Overview

The `optimize-relinearization` pass optimizes the placement of relinearization
operations in FHE circuits by deferring them as late as possible. This
optimization reduces the total number of relinearizations needed and minimizes
noise accumulation, leading to better parameter efficiency and potentially
avoiding expensive bootstrapping operations.

## Input/Output

- **Input**: MLIR module with management dialect operations including
  relinearization operations
- **Output**: Same module with relinearization operations moved to optimal
  positions based on ILP (Integer Linear Programming) analysis

## Options

- `--use-loc-based-variable-names` (bool, default: false): Use operation source
  locations in ILP variable names for debugging
  - Helpful when debugging ILP model construction issues
  - Should be used with IR written to disk and as first/only pass
- `--allow-mixed-degree-operands` (bool, default: true): Allow operations with
  mixed-degree ciphertext inputs
  - Enables operations between ciphertexts with different key bases
  - Supported by most FHE backends (OpenFHE, Lattigo)
  - Disable for backends that require uniform ciphertext degrees

## Usage Examples

```bash
# Optimize relinearization with default settings
heir-opt --optimize-relinearization input.mlir

# Enable location-based debugging (run as standalone pass)
heir-opt --optimize-relinearization --use-loc-based-variable-names input.mlir

# Disable mixed-degree operations for restrictive backends
heir-opt --optimize-relinearization --allow-mixed-degree-operands=false input.mlir

# Use in typical FHE optimization pipeline
heir-opt --operation-balancer --optimize-relinearization --generate-param-bgv input.mlir
```

## When to Use

This pass should be used in FHE compilation pipelines:

1. **After Management Insertion**: Run after passes that insert management
   operations
1. **Before Parameter Generation**: Optimized relinearization affects parameter
   requirements
1. **For Performance Optimization**: Reduces computational overhead and noise
   growth

Typical pipeline position:

```bash
heir-opt --secret-insert-mgmt-bgv --annotate-mgmt --optimize-relinearization --generate-param-bgv
```

## Algorithm: Integer Linear Programming

The pass formulates relinearization optimization as an ILP problem for each
function:

### Variables

- **Binary variables** for each potential relinearization operation placement
- **Degree variables** tracking ciphertext key basis at each operation

### Constraints

- **Return Value Linearization**: All function return values must be linearized
- **Uniform Input Degrees**: Operations require same key basis for all
  ciphertext inputs (unless mixed-degree allowed)
- **Rotation Requirements**: Rotation operations require linearized inputs
- **Dependency Constraints**: Maintain proper operation ordering

### Objective

Minimize the total number of relinearization operations while satisfying all
constraints.

## Optimization Benefits

### Noise Reduction

- **Deferred Relinearization**: Delays noise introduction from relinearization
- **Fewer Operations**: Reduces total relinearization count
- **Better Bootstrapping**: May eliminate need for bootstrapping in some cases

### Performance Improvement

- **Reduced Computation**: Fewer expensive relinearization operations
- **Better Parallelization**: More flexible operation scheduling
- **Lower Memory Usage**: Fewer intermediate linearized ciphertexts

### Parameter Efficiency

- **Smaller Auxiliary Modulus**: Fewer relinearizations need smaller P modulus
- **Reduced Key Size**: Smaller evaluation keys due to fewer relinearizations

## Example Transformation

**Input (Suboptimal):**

```mlir
func.func @dot_product(%a: !secret.secret<tensor<4xi32>>, %b: !secret.secret<tensor<4xi32>>) -> !secret.secret<i32> {
  %0 = secret.generic(%a, %b : !secret.secret<tensor<4xi32>>, !secret.secret<tensor<4xi32>>) {
  ^body(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>):
    %c0 = arith.constant 0 : i32

    // Multiply elements (creates degree-2 ciphertexts)
    %1 = arith.muli %arg0[0], %arg1[0] : i32  // degree 2
    %relin1 = mgmt.relinearize %1             // unnecessary early relinearization

    %2 = arith.muli %arg0[1], %arg1[1] : i32  // degree 2
    %relin2 = mgmt.relinearize %2             // unnecessary early relinearization

    // Add products (could work with degree-2 inputs)
    %3 = arith.addi %relin1, %relin2 : i32    // uses linearized inputs
    %4 = arith.addi %3, %c0 : i32

    secret.yield %4 : i32
  } -> !secret.secret<i32>
  return %0 : !secret.secret<i32>
}
```

**Output (Optimized):**

```mlir
func.func @dot_product(%a: !secret.secret<tensor<4xi32>>, %b: !secret.secret<tensor<4xi32>>) -> !secret.secret<i32> {
  %0 = secret.generic(%a, %b : !secret.secret<tensor<4xi32>>, !secret.secret<tensor<4xi32>>) {
  ^body(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>):
    %c0 = arith.constant 0 : i32

    // Multiply elements (creates degree-2 ciphertexts)
    %1 = arith.muli %arg0[0], %arg1[0] : i32  // degree 2
    %2 = arith.muli %arg0[1], %arg1[1] : i32  // degree 2

    // Add degree-2 ciphertexts directly (more efficient)
    %3 = arith.addi %1, %2 : i32              // still degree 2
    %4 = arith.addi %3, %c0 : i32             // still degree 2

    // Single relinearization at the end (required for return)
    %relin = mgmt.relinearize %4              // necessary for return

    secret.yield %relin : i32
  } -> !secret.secret<i32>
  return %0 : !secret.secret<i32>
}
```

## ILP Model Details

For complete ILP formulation details, see the
[HEIR documentation on relinearization ILP](https://heir.dev/docs/design/relinearization_ilp/).

The model is adapted from Jeremy Kun's blog post on MLIR global optimization and
dataflow analysis, extended for FHE-specific constraints and objectives.

## Debugging ILP Models

When ILP model construction fails:

1. Use `--use-loc-based-variable-names=true`
1. Run as standalone pass on clean IR
1. Write IR to disk before running pass
1. Check for unique operation names and locations

## Backend Compatibility

- **OpenFHE**: Supports mixed-degree operations (default settings work)
- **Lattigo**: Supports mixed-degree operations (default settings work)
- **SEAL**: May require `--allow-mixed-degree-operands=false` depending on
  configuration
- **Custom Backends**: Configure based on backend capabilities

## Integration Notes

Works optimally with:

- **Operation Balancer**: Run operation balancer first to create better tree
  structures
- **Management Annotation**: Requires proper `mgmt.mgmt` attributes
- **Parameter Generation**: Optimized placement improves parameter selection
