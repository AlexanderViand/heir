---
title: Operation Balancer
weight: 16
---

## Overview

The `operation-balancer` pass restructures trees of addition and multiplication
operations to minimize computational depth. By balancing arithmetic expression
trees, this pass improves parallelization opportunities and reduces
multiplication depth, which is crucial for FHE performance and parameter
selection.

## Input/Output

- **Input**: MLIR module with arithmetic operations (add, mul) inside
  `secret.generic` operations
- **Output**: Same module with arithmetic trees restructured to minimize depth
  while preserving mathematical equivalence

## Options

This pass takes no options - it automatically analyzes and balances arithmetic
expression trees.

## Usage Examples

```bash
# Balance arithmetic operations for better parallelization
heir-opt --operation-balancer input.mlir

# Use in FHE optimization pipeline
heir-opt --operation-balancer --optimize-relinearization input.mlir

# Apply before parameter generation to reduce depth requirements
heir-opt --operation-balancer --annotate-mgmt --generate-param-bgv input.mlir
```

## When to Use

This pass should be used in FHE compilation pipelines to:

1. **Reduce Multiplication Depth**: Lower depth reduces FHE parameter
   requirements
1. **Improve Parallelization**: Balanced trees expose more parallelism
   opportunities
1. **Optimize Performance**: Better tree structure can improve execution time

Best applied:

- After high-level optimizations that may create unbalanced trees
- Before parameter generation passes that need accurate depth analysis
- In conjunction with relinearization optimization

## Algorithm

The pass uses depth-first search to analyze arithmetic expression graphs:

### Tree Analysis

- Identifies trees/subgraphs of addition and multiplication operations
- Handles intermediate computations used multiple times as separate subtrees
- Preserves computation boundaries (doesn't optimize across function calls)

### Balancing Strategy

- **Addition Trees**: Restructures to minimize depth while maintaining
  associativity
- **Multiplication Trees**: Balances multiplication chains to reduce depth
- **Mixed Operations**: Handles trees with both additions and multiplications
  appropriately

### Optimization Goals

- Minimize overall expression depth
- Preserve mathematical correctness
- Maintain operation dependencies
- Expose parallelization opportunities

## Mathematical Background

For associative operations like addition and multiplication:

- **Unbalanced**: `((a + b) + c) + d` has depth 3
- **Balanced**: `(a + b) + (c + d)` has depth 2

This depth reduction is particularly important in FHE where:

- Multiplication depth determines parameter sizes
- Deeper circuits require larger ciphertext moduli
- Parallel operations can execute simultaneously

## Example Transformation

**Input (Unbalanced):**

```mlir
func.func @unbalanced_sum(%arg0: !secret.secret<i32>, %arg1: !secret.secret<i32>,
                         %arg2: !secret.secret<i32>, %arg3: !secret.secret<i32>) -> !secret.secret<i32> {
  %0 = secret.generic(%arg0, %arg1, %arg2, %arg3 : !secret.secret<i32>, !secret.secret<i32>, !secret.secret<i32>, !secret.secret<i32>) {
  ^body(%a: i32, %b: i32, %c: i32, %d: i32):
    // Depth 3: ((a + b) + c) + d
    %1 = arith.addi %a, %b : i32
    %2 = arith.addi %1, %c : i32
    %3 = arith.addi %2, %d : i32
    secret.yield %3 : i32
  } -> !secret.secret<i32>
  return %0 : !secret.secret<i32>
}
```

**Output (Balanced):**

```mlir
func.func @balanced_sum(%arg0: !secret.secret<i32>, %arg1: !secret.secret<i32>,
                       %arg2: !secret.secret<i32>, %arg3: !secret.secret<i32>) -> !secret.secret<i32> {
  %0 = secret.generic(%arg0, %arg1, %arg2, %arg3 : !secret.secret<i32>, !secret.secret<i32>, !secret.secret<i32>, !secret.secret<i32>) {
  ^body(%a: i32, %b: i32, %c: i32, %d: i32):
    // Depth 2: (a + b) + (c + d) - can execute in parallel
    %1 = arith.addi %a, %b : i32    // Parallel with %2
    %2 = arith.addi %c, %d : i32    // Parallel with %1
    %3 = arith.addi %1, %2 : i32    // Depends on both %1 and %2
    secret.yield %3 : i32
  } -> !secret.secret<i32>
}
```

## Benefits for FHE

### Parameter Optimization

- **Smaller Moduli**: Reduced depth allows smaller ciphertext moduli
- **Lower Security Requirements**: Shallower circuits need fewer security
  margins
- **Better Bootstrapping**: Fewer levels consumed before bootstrapping needed

### Performance Improvements

- **Parallelization**: Balanced trees expose independent operations
- **Reduced Latency**: Parallel execution reduces overall computation time
- **Memory Efficiency**: Better cache locality from structured access patterns

### Compilation Benefits

- **Accurate Analysis**: Provides accurate depth for parameter generation
- **Optimization Opportunities**: Enables other passes to make better decisions
- **Backend Flexibility**: Backends can exploit parallel structure

## Limitations

- **Scope**: Only optimizes within `secret.generic` operations
- **Intermediate Values**: Preserves computations used multiple times
  (conservative approach)
- **Global Optimization**: Not globally optimal - focuses on local tree
  balancing

## Integration with Other Passes

Works synergistically with:

- **Relinearization Optimization**: Balanced trees improve relinearization
  placement
- **Parameter Generation**: Accurate depth analysis for parameter sizing
- **Backend Lowering**: Structured trees map better to parallel backends

## Research Background

Inspired by section 2.6 of "EVA Improved: Compiler and Extension Library for
CKKS" by Chowdhary et al., which demonstrated the importance of arithmetic tree
balancing for FHE performance optimization.
