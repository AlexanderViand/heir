# Frontend MLIR Intrinsics

**Branch:** alex/frontend-intrinsics **Status:** WIP - Infrastructure ready,
matmul incomplete **Priority:** High - Enables advanced operations

## Summary

Implements infrastructure for using MLIR operations as Python intrinsics (e.g.,
`linalg.matmul`). Allows direct use of MLIR ops from Python frontend.

## What's Done

1. **Intrinsics infrastructure**

   - `@mlir_op` decorator for defining MLIR operations
   - Modified emitter for special handling
   - Initial `linalg.matmul` support

1. **Type system updates**

   - Added tensor support
   - Fixed Numba signature handling
   - Created `linalg` module structure

## What's Left TODO

- Fix noise analysis for floating-point ops (`addf`) ✗
- Complete matmul emission logic (WIP) ✗
- Implement type inference for matmul outputs ✗
- Add more intrinsic operations ✗
- Create proper registry system (not name matching) ✗

## Issues & Review Comments

1. **Floating-point blocker**: Noise analysis breaks with `addf`
1. **Registry system**: Name matching is fragile
1. **Type inference**: Needs work for complex ops
1. **Documentation**: How to add new intrinsics?
1. **Coordination**: Overlaps with frontend-matmul branch

## Recommendations

1. Fix noise analysis issue first (blocking)
1. Implement proper intrinsic registry
1. Complete matmul as proof of concept
1. Document intrinsic addition process
1. Consider merging with frontend-matmul efforts
1. Add more common operations (conv, pooling, etc.)

## Impact

Enables advanced operations in frontend. Critical for ML workloads but blocked
by technical issues.
