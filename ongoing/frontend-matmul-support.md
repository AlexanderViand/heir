# Frontend Matmul Support

**Branch:** alex/frontend-matmul **Status:** WIP - Blocked by MLIR emission
issues **Priority:** High - Critical for ML workloads

## Summary

Adds matrix multiplication support to HEIR frontend. Overlaps significantly with
frontend-intrinsics branch.

## What's Done

1. **Basic structure**

   - `linalg.matmul` operation setup
   - Type inference following NumPy semantics
   - SSA IR debug output

1. **Experiments**

   - Custom function emission attempts
   - Type casting infrastructure (shared with frontend-casts)

## What's Left TODO

- Fix op_name mapping issue (blocking) ✗
- Complete MLIR emission for matmul ✗
- Handle different scenarios (vector-vector, matrix-vector, matrix-matrix) ✗
- Add comprehensive testing ✗
- Integrate with intrinsics framework ✗

## Issues & Review Comments

1. **Op mapping**: Simple name mapping doesn't work for MLIR
1. **Duplication**: Overlaps with frontend-intrinsics work
1. **Incomplete**: Marked as WIP, core functionality missing
1. **Integration**: Needs coordination with intrinsics branch

## Recommendations

1. **Merge with frontend-intrinsics** - They're solving same problem
1. Focus on fixing op_name mapping issue
1. Add tests for various matmul scenarios
1. Document NumPy compatibility level
1. Consider consolidating efforts

## Impact

Critical for ML workloads but currently blocked. Should be consolidated with
intrinsics work.
