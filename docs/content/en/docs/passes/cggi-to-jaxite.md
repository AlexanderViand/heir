---
title: cggi-to-jaxite
weight: 42
---

## Overview

Converts CGGI dialect operations to JAXite library-specific operations, enabling
code generation for the Python/JAX-based FHE implementation optimized for
research and prototyping. This pass generates Python code that leverages JAX's
NumPy-compatible operations and JIT compilation for flexible FHE development.

## Input/Output

- **Input**: IR with CGGI dialect operations (`cggi.lut3`, `cggi.and`, etc.)
- **Output**: IR with JAXite dialect operations that generate JAXite Python API
  calls for boolean circuit evaluation

## Options

This pass does not currently expose command-line options.

## Usage Examples

```bash
# Basic usage for JAXite backend targeting
heir-opt --cggi-to-jaxite input.mlir

# Complete pipeline from secret to JAXite
heir-opt --secret-to-cggi --cggi-to-jaxite input.mlir

# Research pipeline with optimization
heir-opt --yosys-optimizer --secret-to-cggi --cggi-to-jaxite input.mlir
```

## When to Use

- **Research Applications**: When developing and experimenting with FHE
  algorithms
- **Prototyping**: For rapid prototyping of boolean circuit evaluations
- **Python Integration**: When building Python-based FHE applications
- **Interactive Development**: When using Jupyter notebooks for FHE development
- **Machine Learning**: When integrating FHE with ML frameworks

## Prerequisites

- Input should contain CGGI dialect operations
- Should run after `secret-to-cggi` or similar CGGI generation passes
- JAXite library must be available as a Python package
- JAX and NumPy must be installed in the Python environment

## Technical Notes

- **Python Code Generation**: Produces JAXite Python API calls for boolean
  operations
- **JAX Integration**: Leverages JAX's NumPy-compatible array operations and
  automatic differentiation
- **JIT Compilation**: Supports JAX's just-in-time compilation for improved
  performance
- **Research Focus**: Optimized for flexibility and ease of experimentation
  rather than production performance

## Research Benefits

- **Rapid Prototyping**: Quick iteration on FHE algorithm development
- **Interactive Development**: Compatible with Jupyter notebooks and IPython
- **Scientific Ecosystem**: Integration with NumPy, SciPy, and ML frameworks
- **Automatic Differentiation**: Leverages JAX's autodiff capabilities
- **Experimentation**: Easy modification and testing of FHE algorithms

## Python Ecosystem Integration

- **NumPy Compatibility**: Familiar array operations for scientific computing
- **Jupyter Notebooks**: Interactive development and visualization
- **Machine Learning**: Integration with TensorFlow, PyTorch, and other ML
  frameworks
- **Data Science**: Compatible with pandas, matplotlib, and other data tools

## Performance Considerations

- **JIT Compilation**: JAX's JIT can provide significant performance
  improvements
- **Vectorization**: JAX's vectorized operations for efficient computation
- **Research Trade-offs**: Optimized for flexibility over maximum performance
- **Python Overhead**: Some performance trade-offs due to Python interpreter

## Related Passes

- `secret-to-cggi`: Generates CGGI operations (should run before)
- `cggi-to-tfhe-rust`: Alternative backend for production performance
- `yosys-optimizer`: Boolean circuit optimization (may run before)
- `cggi-to-tfhe-rust-bool`: Alternative Rust-based boolean targeting
