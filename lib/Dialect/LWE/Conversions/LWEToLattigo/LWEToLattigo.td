#ifndef LIB_DIALECT_LWE_CONVERSIONS_LWETOLATTIGO_LWETOLATTIGO_TD_
#define LIB_DIALECT_LWE_CONVERSIONS_LWETOLATTIGO_LWETOLATTIGO_TD_

include "mlir/Pass/PassBase.td"

/// **Developer Documentation: LWEToLattigo Pass**
///
/// This pass implements the backend targeting from generic LWE operations to
/// Lattigo library-specific operations, enabling code generation for the
/// Go-based FHE library that emphasizes performance and ease of use.
///
/// **Implementation Details:**
/// - Converts generic LWE operations to Lattigo-specific Go API calls
/// - Handles both BGV and CKKS scheme operations through Lattigo's interface
/// - Manages Lattigo-specific parameter settings and optimizations
/// - Supports tensor operations for efficient SIMD operations
///
/// **Backend Integration:**
/// - Generates code compatible with Lattigo Go API
/// - Handles Lattigo context and parameter management
/// - Supports Lattigo's modular arithmetic optimizations
/// - Integrates with Lattigo's built-in performance features
///
/// **Go Language Features:**
/// - Leverages Go's memory management and garbage collection
/// - Supports Go's concurrency features for parallel FHE operations
/// - Integrates with Go's standard library and ecosystem
/// - Provides idiomatic Go code generation patterns
///
/// **Reusability Patterns:**
/// - Template for other Go-based FHE backends
/// - Uses standard MLIR code generation patterns
/// - Extensible framework for Lattigo API evolution
///
/// **Extension Points:**
/// - Add support for new Lattigo operations as they are introduced
/// - Implement Lattigo-specific optimization patterns
/// - Add support for advanced Lattigo features (multiparty computation, etc.)
/// - Extend integration with Go ecosystem tools
///
/// **Code Generation Features:**
/// - Generates efficient Lattigo Go code
/// - Handles Go-style error handling and resource management
/// - Supports both synchronous and asynchronous operation patterns
/// - Integrates with Lattigo's parameter optimization
///
/// **Performance Considerations:**
/// - Leverages Lattigo's optimized Go implementations
/// - Supports Go's goroutines for parallel computation
/// - Efficient memory usage patterns for Go runtime
/// - Compatible with Lattigo's vectorization features
///
/// **Dependencies:**
/// - Requires LWE, Lattigo, BGV, and Tensor dialects
/// - Generates code that imports Lattigo Go packages
/// - Compatible with Go build toolchain and module system
def LWEToLattigo : Pass<"lwe-to-lattigo"> {
  let summary = "Lower `lwe` to `lattigo` dialect.";

  let description = [{
    This pass lowers the `lwe` dialect to `lattigo` dialect, enabling code
    generation for the Lattigo library, a high-performance Go-based FHE
    implementation that emphasizes ease of use and integration with the Go
    ecosystem.

    The pass converts generic LWE operations to Lattigo-specific Go API calls,
    supporting both BGV and CKKS schemes through Lattigo's unified interface.
    It generates idiomatic Go code that leverages Lattigo's optimizations and
    integrates well with Go's concurrency features.

    **Technical Notes:**
    - Generates Lattigo Go API calls with proper error handling
    - Supports Go's goroutines for parallel FHE computation
    - Handles Lattigo context creation and parameter management
    - Compatible with tensor operations for SIMD-packed operations
    - Integrates with Go's module system and build toolchain

    **Go Language Benefits:**
    - Leverages Go's garbage collection for memory management
    - Supports concurrent FHE operations through goroutines
    - Provides clean, readable generated code
    - Integrates with Go's rich standard library and ecosystem
  }];

  let dependentDialects = [
    "mlir::heir::bgv::BGVDialect",
    "mlir::heir::lattigo::LattigoDialect",
    "mlir::tensor::TensorDialect",
    "mlir::heir::lwe::LWEDialect",
  ];
}

#endif  // LIB_DIALECT_LWE_CONVERSIONS_LWETOLATTIGO_LWETOLATTIGO_TD_
