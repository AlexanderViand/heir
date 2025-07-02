# Skip Arithmetization Option

**Branch:** alex/no-arithmetization-pipeline **Status:** Complete and ready for
merge **Priority:** Medium - Advanced user feature

## Summary

Adds option to skip arithmetization pipeline for advanced use cases where input
is already properly arithmetized and packed.

## What's Done

1. **New flag**

   - `--enable-arithmetization` (default true)
   - When false, only runs minimal pipeline:
     - `wrapGeneric`
     - `addClientInterface` (without layout)

1. **OpenFHE emitter update**

   - `--skip-vector-resizing` option
   - Skips resize/cyclical padding

1. **Test example**

   - `custom_arithmetization` example
   - Shows advanced use case

## What's Left TODO

- None - implementation complete

## Issues & Review Comments

1. **Documentation**: Could use more explanation
1. **Use cases**: Example could have more comments
1. **Testing**: Good test coverage

## Recommendations

1. **Ready to merge** after minor doc additions
1. Add documentation explaining when to use
1. Expand example with detailed comments
1. Document what "properly arithmetized" means
1. Add to advanced user guide

## Impact

Enables advanced users to bypass standard pipeline. Useful for custom
arithmetization strategies.
