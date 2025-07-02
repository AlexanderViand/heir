# Data-Oblivious Benchmarks

**Branch:** alex/data-oblivious-bmks **Status:** Partially complete - Set
Intersection done, others WIP **Priority:** High - Security-critical feature

## Summary

Implements data-oblivious versions of VIP-Bench algorithms to prevent timing
side-channel attacks. Creates both native and data-oblivious implementations for
comparison.

## What's Done

1. **Set Intersection** ✓ Complete

   - Native version with conditional branches (`scf.if`)
   - Data-oblivious with `arith.select` (CMOV-style)
   - Full LLVM lowering and tests

1. **Infrastructure**

   - Bazel build setup
   - Test runners
   - OpenFHE integration
   - Pattern established for other benchmarks

1. **Started implementations**

   - Distinctness (WIP)
   - Boyer-Moore Search (basic structure)
   - Simple-if, sum_tensor, gcd_list (patterns)

## What's Left TODO

- Complete Distinctness implementation
- Fix Simple-If (has error.txt)
- Add dynamic size support (hardcoded n:256, m:3, SIZE:20)
- Change argument types to secret types
- Integrate MLIR-to-LLVM directly (remove shell scripts)
- Support cyclic repetition in add-client-interface (#645)
- Replace setup with HEIR-generated helpers (#661)

## Issues & Review Comments

1. **Hardcoded values**: Should be parameterizable
1. **Incomplete implementations**: Several benchmarks unfinished
1. **Test infrastructure**: Shell scripts should be replaced
1. **Documentation**: Needs explanation of data-oblivious patterns

## Recommendations

1. Finish Distinctness implementation first
1. Fix Simple-If errors
1. Parameterize all hardcoded values
1. Add documentation explaining oblivious patterns
1. Create performance comparison framework
1. Add security analysis/proofs

## Impact

Critical for security - prevents timing attacks in FHE applications. Set
Intersection provides good template for other algorithms.
