---
title: CGGI Boolean Vectorize
weight: 10
---

## Overview

The `cggi-boolean-vectorize` pass groups independent logic gates into packed
operations to improve performance through batching. This pass combines different
types of boolean gates (not just identical operations) into single calls using
packed APIs, optimizing for backends that support batch processing of boolean
operations.

## Input/Output

- **Input**: MLIR module with individual CGGI boolean gate operations
- **Output**: Same module with independent operations grouped into packed batch
  operations

## Options

- `--parallelism` (int, default: 0): Parallelism factor for batching operations
  - 0 indicates infinite parallelism (batch all possible operations)
  - Positive values limit the batch size to the specified number

## Usage Examples

```bash
# Group all independent boolean gates with unlimited parallelism
heir-opt --cggi-boolean-vectorize input.mlir

# Limit batching to groups of 8 operations
heir-opt --cggi-boolean-vectorize --parallelism=8 input.mlir

# Use specific parallelism factor for target hardware
heir-opt --cggi-boolean-vectorize --parallelism=16 input.mlir
```

## When to Use

This pass should be used in CGGI-based boolean circuit compilation pipelines:

1. After CGGI operations have been generated from higher-level boolean logic
1. Before lowering to specific backend implementations
1. When targeting backends that support packed/batch boolean operations (e.g.,
   TFHE-rs FPT API)

Particularly beneficial for:

- Circuits with many independent boolean operations
- Backends with efficient packed gate implementations
- Reducing function call overhead in generated code

## Backend Integration

This pass is specifically designed for the TFHE-rs FPT (Fast Packed Transform)
API, where packed operations take the form:

```rust
let outputs_ct = fpga_key.packed_gates(&gates, &ref_to_ct_lefts, &ref_to_ct_rights);
```

The pass groups gates into:

- A vector of gate type specifications
- Left operand ciphertext vectors
- Right operand ciphertext vectors

## Algorithm

The pass uses analysis similar to straight-line vectorization but with key
differences:

- Combines different types of boolean gates (AND, OR, XOR, etc.)
- Not restricted to identical operations
- Focuses on independence rather than operation similarity
- Optimizes for packed API call patterns

## Example Transformation

**Input:**

```mlir
func.func @independent_gates(%a: !lwe.lwe_ciphertext, %b: !lwe.lwe_ciphertext,
                           %c: !lwe.lwe_ciphertext, %d: !lwe.lwe_ciphertext) ->
                           (!lwe.lwe_ciphertext, !lwe.lwe_ciphertext) {
  %0 = cggi.and %a, %b : !lwe.lwe_ciphertext
  %1 = cggi.xor %c, %d : !lwe.lwe_ciphertext
  return %0, %1 : !lwe.lwe_ciphertext, !lwe.lwe_ciphertext
}
```

**Output:**

```mlir
func.func @independent_gates(%a: !lwe.lwe_ciphertext, %b: !lwe.lwe_ciphertext,
                           %c: !lwe.lwe_ciphertext, %d: !lwe.lwe_ciphertext) ->
                           (!lwe.lwe_ciphertext, !lwe.lwe_ciphertext) {
  %gates = arith.constant ["and", "xor"] : vector<2x!StringAttr>
  %left_ops = tensor.from_elements %a, %c : tensor<2x!lwe.lwe_ciphertext>
  %right_ops = tensor.from_elements %b, %d : tensor<2x!lwe.lwe_ciphertext>
  %results = cggi.packed_gates %gates, %left_ops, %right_ops :
    (vector<2x!StringAttr>, tensor<2x!lwe.lwe_ciphertext>, tensor<2x!lwe.lwe_ciphertext>) ->
    tensor<2x!lwe.lwe_ciphertext>
  %0 = tensor.extract %results[0] : tensor<2x!lwe.lwe_ciphertext>
  %1 = tensor.extract %results[1] : tensor<2x!lwe.lwe_ciphertext>
  return %0, %1 : !lwe.lwe_ciphertext, !lwe.lwe_ciphertext
}
```

## Performance Benefits

- **Reduced Function Call Overhead**: Multiple operations handled in single
  backend call
- **Improved Memory Locality**: Batch processing of related data
- **Backend Optimization**: Enables backend-specific optimizations for packed
  operations
- **Parallelization**: Backends can process multiple gates simultaneously
