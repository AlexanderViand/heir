# HEIR Branch Review Summary

This directory contains detailed reviews of all branches on the `alex` remote.
Each file provides:

- Branch status and priority
- Summary of goals
- What's implemented
- What remains TODO
- Issues and review comments
- Recommendations

## Branch Categories

### Ready to Merge

1. **ai-documentation.md** - Comprehensive documentation improvements
1. **skip-arithmetization-option.md** - Advanced user feature for custom
   arithmetization
1. **bazel-openfhe-integration.md** - Developer experience improvement

### Needs Refactoring Separation

These branches mix new features with massive refactoring that should be split:

1. **cggi-rust-backend.md** - New tfhe-rs backend (mixed with cleanup)
1. **lwe-types-migration.md** - Infrastructure improvement (mixed with cleanup)
1. **openfhe-to-scheme-pass.md** - Critical abstraction layer (mixed with
   cleanup)

### Active Development - High Priority

1. **data-oblivious-benchmarks.md** - Security-critical feature (Set
   Intersection done, others WIP)
1. **frontend-type-casting.md** - Major usability improvement (complete, needs
   rebase)
1. **rlwe-polynomial-lowering.md** - Core infrastructure (complex, significant
   work remaining)

### Overlapping Efforts (Should Merge)

1. **frontend-mlir-intrinsics.md** - Infrastructure for MLIR ops in Python
1. **frontend-matmul-support.md** - Matmul implementation (overlaps with
   intrinsics)

### Needs Polish

1. **heracles-backend.md** - Performance analysis backend (feature-complete,
   needs fixes)

### Early Stage

1. **pisa-hardware-dialect.md** - Hardware acceleration (very early, needs work)
1. **arithmetic-documentation.md** - Design docs (barely started)

## Key Recommendations

1. **Separate refactoring**: Extract the large-scale cleanup into its own PR
1. **Merge overlapping work**: Combine frontend-intrinsics and frontend-matmul
1. **Focus on blockers**: Fix noise analysis for floating-point ops
1. **Complete in-progress work**: Finish data-oblivious benchmarks
1. **Rebase old branches**: Many branches need rebasing on main

## Priority Order

1. Merge ready branches (ai-documentation, skip-arithmetization, bazel-openfhe)
1. Extract and merge refactoring
1. Complete data-oblivious benchmarks
1. Fix frontend intrinsics blocker
1. Complete RLWE lowering
1. Polish Heracles backend
1. Continue PISA development
