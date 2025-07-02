# Heracles SDK Backend

**Branch:** alex/heracles_demo **Status:** Feature-complete but needs polish
**Priority:** Medium - New backend for performance analysis

## Summary

Adds Heracles SDK backend for BGV/CKKS with instruction-level tracing. Enables
detailed performance analysis of FHE operations.

## What's Done

1. **Complete Heracles backend**

   - SDK emitter for BGV/CKKS operations
   - Instruction-level CSV output for performance analysis
   - Protobuf support for data serialization
   - Python bindings via pybind11
   - Integration with HEIR's Python frontend

1. **Supported operations**

   - Arithmetic: add, sub, mul
   - Advanced: rotate, rescale, relinearize
   - Data helper emitters for I/O

1. **Test infrastructure**

   - emit_heracles_sdk tests
   - Integration examples

## What's Left TODO

- Fix "ImportError: generic_type" with double setup (#1162)
- Rethink server/client split (#1119)
- Expose ciphertext serialization in Python
- Remove unregistered dialect dependency (#1414)
- Implement negate instruction (uses mul_plain with -1)
- Add dimension checking for operations
- Improve entry function heuristics
- Implement `lwe.reinterpret_underlying_type`

## Issues & Review Comments

1. **FIXMEs**: Several inefficient implementations
1. **Error handling**: Missing dimension mismatch checks
1. **Hardcoded assumptions**: Key/ciphertext parameters
1. **Entry function**: Detection needs improvement
1. **Negate workaround**: Should have native support

## Recommendations

1. Fix ImportError issue first (blocking)
1. Add proper error handling for dimensions
1. Implement native negate operation
1. Improve entry function detection
1. Add performance analysis examples
1. Document CSV output format

## Impact

Enables detailed performance profiling of FHE computations. Valuable for
optimization and debugging.
