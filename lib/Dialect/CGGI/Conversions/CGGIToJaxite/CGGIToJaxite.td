#ifndef LIB_DIALECT_CGGI_CONVERSIONS_CGGITOJAXITE_CGGITOJAXITE_TD_
#define LIB_DIALECT_CGGI_CONVERSIONS_CGGITOJAXITE_CGGITOJAXITE_TD_

include "mlir/Pass/PassBase.td"

/// **Developer Documentation: CGGIToJaxite Pass**
///
/// This pass implements backend targeting from CGGI boolean operations to
/// JAXite library-specific operations, enabling code generation for the
/// Python/JAX-based FHE library that emphasizes research and prototyping.
///
/// **Implementation Details:**
/// - Converts CGGI boolean operations to JAXite Python API calls
/// - Handles boolean circuits and lookup table operations
/// - Integrates with JAX's automatic differentiation and JIT compilation
/// - Supports research-oriented FHE development workflows
///
/// **JAXite Integration:**
/// - Generates code compatible with JAXite Python API
/// - Leverages JAX's NumPy-compatible array operations
/// - Supports JAX's just-in-time compilation for performance
/// - Integrates with Python's scientific computing ecosystem
///
/// **Research Focus:**
/// - Optimized for research and prototyping workflows
/// - Compatible with Jupyter notebooks and interactive development
/// - Supports rapid experimentation with FHE algorithms
/// - Enables integration with machine learning frameworks
def CGGIToJaxite : Pass<"cggi-to-jaxite"> {
  let summary = "Lower `cggi` to `jaxite` dialect.";

  let description = [{
    This pass lowers the `cggi` dialect to `jaxite` dialect, enabling
    code generation for the JAXite library, a Python/JAX-based FHE
    implementation optimized for research and prototyping.

    The pass converts CGGI boolean operations to JAXite-specific Python
    API calls, supporting boolean circuit evaluation with the flexibility
    and ease of use of the Python ecosystem.

    **Technical Notes:**
    - Generates JAXite Python API calls for boolean operations
    - Leverages JAX's NumPy-compatible array operations
    - Supports JAX's JIT compilation for improved performance
    - Integrates with Python scientific computing ecosystem

    **Research Benefits:**
    - Rapid prototyping and experimentation capabilities
    - Interactive development through Jupyter notebooks
    - Integration with machine learning and data science tools
    - Flexible Python-based development environment
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::heir::jaxite::JaxiteDialect",
  ];
}

#endif  // LIB_DIALECT_CGGI_CONVERSIONS_CGGITOJAXITE_CGGITOJAXITE_TD_
