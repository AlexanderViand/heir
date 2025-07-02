---
title: Lower Unpack
weight: 16
---

## Overview

The `lower-unpack` pass converts HEIR-specific `tensor_ext.unpack` operations to
equivalent standard MLIR operations using only the tensor and arithmetic
dialects. This lowering enables broader compatibility with standard MLIR
optimization passes.

## Input/Output

- **Input**: MLIR IR containing `tensor_ext.unpack` operations from the HEIR
  tensor extension dialect
- **Output**: MLIR IR using only standard tensor and arithmetic operations with
  equivalent semantics

## Options

This pass takes no command-line options.

## Usage Examples

```bash
# Lower tensor_ext.unpack operations to standard MLIR
heir-opt --lower-unpack input.mlir

# Common usage before standard MLIR optimization passes
heir-opt --lower-unpack --canonicalize input.mlir
```

## When to Use

This pass is essential when:

1. **Compatibility with standard MLIR**: Before applying standard MLIR
   optimization passes that don't recognize HEIR extensions
1. **Lowering pipeline**: As part of progressively lowering HEIR-specific
   operations to standard representations
1. **Cross-dialect optimization**: To enable optimizations that work across
   tensor and arithmetic dialects
1. **Backend preparation**: Before targeting backends that only support standard
   MLIR operations
1. **Tool integration**: When using MLIR tools that expect only standard
   dialects

The pass acts as a bridge between HEIR-specific abstractions and the broader
MLIR ecosystem.

## Implementation Details

The transformation process:

1. **Operation Detection**: Identifies all `tensor_ext.unpack` operations in the
   IR
1. **Semantic Analysis**: Analyzes the unpacking semantics to determine
   equivalent standard operations
1. **Standard Translation**: Converts to equivalent sequences of standard tensor
   and arithmetic operations
1. **Type Consistency**: Ensures type compatibility throughout the
   transformation

**Design Goals:**

- **Semantic Preservation**: Maintains exact computational semantics during
  lowering
- **Standard Compliance**: Uses only operations from standard MLIR dialects
- **Optimization Enablement**: Creates opportunities for standard MLIR
  optimization passes
- **Performance Neutrality**: Avoids performance degradation during the lowering
  process

**Benefits:**

- Enables use of the full MLIR optimization ecosystem
- Provides compatibility with downstream tools and backends
- Simplifies the overall compilation pipeline by reducing custom operations
- Facilitates integration with existing MLIR-based workflows

**Integration Patterns:**

- Typically used after HEIR-specific optimizations
- Often followed by standard MLIR canonicalization and optimization passes
- Part of the progressive lowering strategy from high-level HEIR abstractions to
  target-specific code
