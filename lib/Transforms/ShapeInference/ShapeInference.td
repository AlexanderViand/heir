#ifndef LIB_TRANSFORMS_SHAPEINFERENCE_SHAPEINFERENCE_TD_
#define LIB_TRANSFORMS_SHAPEINFERENCE_SHAPEINFERENCE_TD_

include "mlir/Pass/PassBase.td"

// Pass for shape inference using dataflow analysis with type interfaces.
// This resolves dynamic shapes to static shapes by propagating shape information
// from annotated function arguments through operations implementing InferTypeOpInterface.
//
// IMPLEMENTATION DETAILS:
// - Forward dataflow analysis starting from shape.shape annotations
// - InferTypeOpInterface integration for automatic shape propagation
// - Type updating for shaped types with resolved dimensions
// - Function argument annotation processing
//
// REUSABILITY PATTERNS:
// - Interface-based design allows extension to custom operations
// - Annotation-driven approach can be adapted for other type inference problems
// - Dataflow framework can be reused for other type analysis passes
// - Can be extended to support more sophisticated shape relationships
//
// DESIGN CONSIDERATIONS:
// - Designed for Python frontend integration where ranks are known
// - Relies on MLIR's standard type inference interfaces
// - Focused on static shape resolution for optimization enablement
// - Assumes shape annotations provide accurate dimension information
def ShapeInference : Pass<"shape-inference"> {
  let summary = "Infer shapes for shaped types";

  let description = [{
    This pass infers the shapes of shaped types in a function,
    starting from function arguments annotated with a {shape.shape} attribute.
    Shape inference is only supported for operations that implement InferTypeOpInterface.

    This is primarily intended to be used in conjunction with the Python frontend,
    which infers the rank, but not the length of each dimension, for tensor types.
  }];

  let dependentDialects = [
  ];
}


#endif // LIB_TRANSFORMS_SHAPEINFERENCE_SHAPEINFERENCE_TD_
