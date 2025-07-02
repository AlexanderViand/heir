# RLWE Polynomial Lowering

**Branch:** alex/rlwe_kernels **Status:** Complex WIP - Significant work
remaining **Priority:** High - Core infrastructure

## Summary

Implements lowering of RLWE operations (particularly relinearization) to
polynomial-level operations. Critical for backend flexibility.

## What's Done

1. **BGV to Polynomial conversion**

   - Basic ops: add, sub, negate, mul
   - Partial relinearization (dimension 3→2 only)
   - Uses new helper ops for key switching

1. **New LWE/PolyExt helper ops**

   - `lwe.key_switching_key` - placeholder for keys
   - `poly_ext.gadget_product` - noise-optimized product
   - `poly_ext.cmod_switch` - modulus switching
   - `poly_ext.ksk_delta` - key switching delta
   - `poly_ext.digit_decompose` - decomposition

1. **Started PolyExtToPolynomial pass**

   - Framework in place
   - WIP `ConvertGadgetProduct`

## What's Left TODO

### BGVToPolynomial

- General multiplication for dimension > 2 (TODO #999999) ✗
- General relinearization beyond 3→2 ✗
- Proper type/parameter verification ✗

### PolyExtToPolynomial

- Complete `ConvertGadgetProduct`: ✗
  - Make modulus extension toggleable
  - Implement digit decomposition
  - Get plaintext modulus properly
  - Add delta correction factor
- Implement lowerings for: ✗
  - `cmod_switch`
  - `ksk_delta`
  - `digit_decompose`

### General

- Pass to lift key switching keys to function parameters ✗
- Support for key generation in backends ✗
- OpenFHE key switching key integration ✗
- Revert local LLVM repo modification ✗

## Issues & Review Comments

1. **CRITICAL**: Uses local LLVM repo - must revert
1. **Hardcoded values**: modulus 12345, plaintext mod 42
1. **Limited scope**: Only handles dimension 2/3
1. **Fake TODO number**: #999999 should be real issue
1. **Type inference**: Missing for new ops
1. **Incomplete lowerings**: Most PolyExt ops not implemented

## Recommendations

1. **Revert LLVM change immediately**
1. Replace hardcoded values with proper sources
1. Complete dimension 2 implementation first
1. Then generalize to arbitrary dimensions
1. Add comprehensive tests for each op
1. Document polynomial representation clearly
1. Create real issue numbers for TODOs

## Impact

Critical infrastructure for flexible backend targeting. Blocks full BGV support
without completion.
