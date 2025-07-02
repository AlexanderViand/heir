# PISA Hardware Dialect

**Branch:** alex/pisa_dialect **Status:** Early development - single WIP commit
**Priority:** Low - Experimental hardware targeting

## Summary

Creates PISA (Programmable Instruction Set Architecture) dialect for polynomial
arithmetic targeting specialized FHE hardware accelerators.

## What's Done

1. **PISA dialect definition**

   - Basic arithmetic: add, sub, mul, muli (immediate)
   - MAC operations: mac, maci
   - NTT/INTT operations (commented out)

1. **Type constraints**

   - Fixed tensor size (8192)
   - 32-bit modular arithmetic

1. **Basic infrastructure**

   - PISA emitter for textual representation
   - Polynomial-to-PISA conversion pass
   - Test skeleton

## What's Left TODO

- Add RNS (Residue Number System) support ✗
- Implement pass to split polynomials > 8k degree ✗
- Fix mod_arith.constant for tensors ✗
- Handle twiddle factors for NTT ✗
- Proper constant/immediate handling ✗
- Avoid duplicating inputs for metadata ✗
- Handle double I/O from CSV format ✗
- Re-enable end-to-end tests (#1199) ✗

## Issues & Review Comments

1. **Very early stage**: Single WIP commit
1. **Many broken features**: NTT commented out, constants broken
1. **Fixed polynomial size**: Limited to 8192
1. **No RNS**: Critical for efficient FHE
1. **Missing documentation**: No explanation of PISA ISA
1. **Test infrastructure**: Tests disabled

## Recommendations

1. **Priority**: This needs significant work before being useful
1. Fix mod_arith.constant issue first
1. Add RNS support (critical for performance)
1. Document PISA ISA and hardware assumptions
1. Enable NTT/INTT operations
1. Add examples of generated PISA code
1. Consider if fixed 8192 size is too limiting

## Impact

Potentially valuable for hardware acceleration but needs substantial
development. Currently too early to be useful.
