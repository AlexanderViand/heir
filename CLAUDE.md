# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Repository Overview

HEIR (Homomorphic Encryption Intermediate Representation) is an MLIR-based
toolchain for homomorphic encryption compilers. The project compiles high-level
programs into optimized FHE (Fully Homomorphic Encryption) implementations
across multiple backends.

## Build System and Commands

HEIR uses **Bazel** as the primary build system with CMake as an alternative.

### Core Build Commands

```bash
# Build all targets
bazel build //...

# Build with CI configuration (recommended)
bazel build --noincompatible_strict_action_env --//:enable_openmp=0 -c fastbuild //...

# Build main tools
bazel build //tools:heir-opt      # Main optimization tool
bazel build //tools:heir-translate  # Code generation tool
```

### Testing Commands

```bash
# Run all tests
bazel test //...

# Run tests with CI configuration
bazel test --noincompatible_strict_action_env --//:enable_openmp=0 -c fastbuild //...

# Run Rust backend tests
bash .github/workflows/run_rust_tests.sh

# Run specific test suites
bazel test //tests/...
bazel test //frontend/...
```

### Configuration Flags

- `--//:enable_openmp=0/1` - OpenMP support (default: disabled)
- `--//:enable_yosys=0/1` - Yosys circuit optimizer (default: enabled)
- `-c fastbuild/dbg/opt` - Build mode

### Python Frontend

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run Python tests
python -m pytest scripts/
```

## Architecture Overview

HEIR follows standard MLIR patterns with FHE-specific extensions:

### Core Tools

- **heir-opt**: Main transformation tool (like mlir-opt) that applies
  optimization passes
- **heir-translate**: Code generation tool that emits target-specific code
- **heir-lsp**: Language server for IDE integration

### Key Components

#### 1. MLIR Dialects (`lib/Dialect/`)

FHE scheme dialects:

- **Secret**: High-level secret types and operations
- **BGV/BFV/CKKS**: RLWE-based schemes for packed arithmetic
- **CGGI**: Boolean circuits for TFHE-style FHE
- **LWE**: Low-level lattice operations

Backend dialects:

- **Openfhe**, **Lattigo**, **TfheRust**, **Jaxite**: Target-specific
  representations

Utility dialects:

- **Polynomial**: Polynomial arithmetic and NTT operations
- **ModArith**: Modular arithmetic
- **TensorExt**: Tensor operations with rotation/layout

#### 2. Transformations (`lib/Transforms/`)

Core optimization passes:

- **Secretize**: Convert plaintext ops to secret equivalents
- **ConvertToCiphertextSemantics**: Layout assignment and scheme lowering
- **LayoutOptimization/LayoutPropagation**: SIMD vectorization optimization
- **YosysOptimizer**: Boolean circuit optimization via Yosys

#### 3. Code Generation (`lib/Target/`)

Emitters for each backend:

- **OpenFhePke**: C++ with OpenFHE library
- **TfheRust**: Rust with TFHE-rs
- **Lattigo**: Go with Lattigo
- **Jaxite**: Python with JAX

#### 4. Pipelines (`lib/Pipelines/`)

End-to-end compilation pipelines:

- `mlir-to-secret-arithmetic`: Standard MLIR → Secret dialect
- `mlir-to-bgv/bfv/ckks`: RLWE scheme pipelines
- `mlir-to-cggi`: Boolean circuit pipeline
- `scheme-to-{openfhe,lattigo,tfhe-rs}`: Backend code generation

### Compilation Flow

1. **Frontend**: Python/MLIR → Standard MLIR dialects
1. **Secretization**: Standard ops → Secret dialect
1. **Scheme Lowering**: Secret → FHE scheme (BGV/CGGI/etc.)
1. **Backend Lowering**: Scheme → Target dialect
1. **Code Generation**: Target dialect → Final code

## Development Workflow

### Running Single Tests

```bash
# Test specific MLIR file
bazel test //tests/Transforms/secretize:secretize.mlir.test

# Test specific example
bazel test //tests/Examples/openfhe:simple_sum_test
```

### Code Generation Commands

```bash
# Apply optimization passes
bazel run //tools:heir-opt -- --pass-pipeline="builtin.module(func.func(secretize))" input.mlir

# Generate code for backend
bazel run //tools:heir-translate -- --emit-openfhe-pke input.mlir
```

### Working with Specific Backends

#### OpenFHE

```bash
# Test OpenFHE examples
bazel test //tests/Examples/openfhe/...
```

#### TFHE-rs

```bash
# Requires Rust toolchain
rustup toolchain install stable
bazel test //tests/Examples/tfhe_rust/...
```

### Development Dependencies

- **MLIR/LLVM**: Automatically handled by Bazel
- **Yosys/ABC**: Optional for circuit optimization
- **OpenFHE**: Downloaded by Bazel for OpenFHE backend
- **Rust**: Required for TFHE-rs backend testing

## File Organization Patterns

- **Dialect structure**: Each dialect has `IR/`, `Transforms/`, `Conversions/`
  subdirectories
- **Test organization**: Tests mirror `lib/` structure in `tests/`
- **Examples**: `tests/Examples/{backend}/` contains end-to-end examples
- **Build files**: `BUILD` (Bazel) and `CMakeLists.txt` in each directory

## Python Frontend (`frontend/`)

The Python frontend provides a Numba-based interface for writing FHE programs:

```python
from heir import compile_function
@compile_function
def my_function(x):
    return x + 1
```

Integration testing uses both Python and MLIR components.
