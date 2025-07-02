---
title: mod-arith-to-arith
weight: 53
---

## Overview

Converts modular arithmetic operations from the `mod_arith` dialect back to
standard arithmetic operations using the `arith` dialect. This pass provides the
reverse transformation from modular arithmetic, enabling integration with
standard MLIR arithmetic optimizations and analyses.

## Input/Output

- **Input**: IR with modular arithmetic operations (`mod_arith.add`,
  `mod_arith.mul`, etc.)
- **Output**: IR with standard arithmetic operations (`arith.addi`,
  `arith.muli`, etc.) that implement the modular arithmetic semantics

## Options

This pass does not currently expose command-line options.

## Usage Examples

```bash
# Basic usage
heir-opt --mod-arith-to-arith input.mlir

# Reverse conversion after modular arithmetic processing
heir-opt --arith-to-mod-arith --find-mac --mod-arith-to-arith input.mlir

# Integration with standard optimizations
heir-opt --mod-arith-to-arith --canonicalize --cse input.mlir
```

## When to Use

- **Standard Optimization**: When applying standard MLIR arithmetic
  optimizations after modular arithmetic processing
- **Backend Integration**: When targeting backends that expect standard
  arithmetic operations
- **Analysis Passes**: Before running arithmetic analysis passes that work on
  standard `arith` operations
- **Reverse Pipeline**: As part of a pipeline that temporarily uses modular
  arithmetic

## Prerequisites

- Input should contain modular arithmetic operations from the `mod_arith`
  dialect
- Often used after modular arithmetic processing or optimization passes
- Compatible with operations that have been processed through
  `arith-to-mod-arith`

## Technical Notes

- **Semantic Preservation**: Maintains the modular arithmetic semantics using
  standard arithmetic operations
- **Pattern-Based Lowering**: Uses declarative rewrite rules (DRR) for
  conversion patterns
- **Operation Mapping**: Direct mapping from modular operations to their
  standard arithmetic equivalents
- **Overflow Handling**: Preserves modular arithmetic overflow behavior

## Implementation Details

- **DRR Patterns**: Uses MLIR's Declarative Rewrite Rules for pattern-based
  conversion
- **Specific Operations**: Includes specialized patterns for operations like
  `mod_arith.subifge`
- **Standard Integration**: Results are compatible with standard MLIR arithmetic
  dialect

## Example Patterns

- **SubIfGE Operation**: Converts conditional subtraction operations to standard
  arithmetic with comparisons and selects
- **Direct Mapping**: Most operations have direct equivalents in standard
  arithmetic

## Related Passes

- `arith-to-mod-arith`: Reverse conversion to modular arithmetic
- `find-mac`: Multiply-accumulate optimization (may run between conversions)
- `canonicalize`: Standard optimization (may run after)
- `cse`: Common subexpression elimination (may run after)
