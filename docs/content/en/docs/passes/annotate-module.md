---
title: Annotate Module
weight: 3
---

## Overview

The `annotate-module` pass annotates the top-level ModuleOp with FHE scheme
and/or backend information. This pass should be called before all lowering
transformations to enable scheme-specific and backend-specific code generation.

## Input/Output

- **Input**: MLIR module without scheme/backend annotations
- **Output**: MLIR module with `scheme.*` and/or `backend.*` attributes on the
  module operation

## Options

| Option    | Type   | Default | Description                                        |
| --------- | ------ | ------- | -------------------------------------------------- |
| `scheme`  | string | `""`    | The FHE scheme to annotate (bgv, ckks, bfv, cggi)  |
| `backend` | string | `""`    | The backend library to annotate (openfhe, lattigo) |

## Usage Examples

```bash
# Annotate with both scheme and backend
heir-opt --annotate-module="backend=openfhe scheme=ckks" input.mlir

# Annotate with only scheme
heir-opt --annotate-module="scheme=bgv" input.mlir

# Annotate with only backend
heir-opt --annotate-module="backend=lattigo" input.mlir
```

### Example Input/Output

**Input:**

```mlir
module {
  func.func @example() {
    return
  }
}
```

**Output:**

```mlir
module attributes {backend.openfhe, scheme.ckks} {
  func.func @example() {
    return
  }
}
```

## When to Use

This pass should be used:

1. **Early in compilation pipeline**: Before any lowering passes that depend on
   scheme/backend selection
1. **Target specification**: To specify the target FHE scheme and implementation
   library
1. **Conditional lowering**: To enable scheme-specific optimizations and code
   generation
1. **Backend selection**: To choose between different FHE library backends

## Available Schemes

- **`bgv`**: Brakerski-Gentry-Vaikuntanathan scheme for integer arithmetic
- **`ckks`**: Cheon-Kim-Kim-Song scheme for approximate real number arithmetic
- **`bfv`**: Brakerski/Fan-Vercauteren scheme for integer arithmetic
- **`cggi`**: Chillotti-Gama-Georgieva-Izabachène scheme for boolean circuits

## Available Backends

- **`openfhe`**: OpenFHE library backend
- **`lattigo`**: Lattigo library backend

## Implementation Details

The pass adds module attributes that:

- Guide subsequent lowering passes in their transformation decisions
- Enable backend-specific optimizations and code generation
- Allow conditional compilation based on scheme capabilities
- Provide metadata for target code generation
