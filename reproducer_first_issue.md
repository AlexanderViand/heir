# Reproducer for issue #1 (ContextAwareDialectConversion rollback leak)

## Prerequisites

- Build **heir-opt** in *debug* mode ("assertions enabled") and link it with
  ASAN/UBSAN to observe the failure clearly.

## Steps

1. Save the MLIR below as `fail_conversion.mlir`.

```mlir
// This module purposely mixes a legal and an illegal type so that
// `--context-aware-dialect-conversion='target-dialect=scf'` fails after it has
// already cloned an scf.for operation (which owns a region) – the exact point
// where `CreateOperationRewrite::rollback()` is invoked.

module {
  func.func @foo(%arg0 : i32, %arg1 : f32) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index

    // The loop body returns an `i32`, but the iter_arg is `f32`.  This is
    // enough to make the type–converter give up *after* materialisations and
    // region-cloning happened.
    %r = scf.for %i = %c0 to %c4 step %c4 iter_args(%x = %arg1) -> (f32) {
      %v = arith.addi %arg0, %i : index // illegal op for the chosen target
      %v2 = arith.sitofp %v : index to f32
      scf.yield %v2 : f32
    }

    return
  }
}
```

2. Run the pass that exercises the **ContextAwareDialectConversion** utility
   (any other pass that uses it will also work):

```bash
heir-opt fail_conversion.mlir \
  --context-aware-dialect-conversion="target-dialect=scf" -o /dev/null
```

## Expected behaviour

The command should fail gracefully with a diagnostic such as
`failure: unable to legalise op`. **Instead**, when compiled with ASAN/UBSAN you
will see a *use-after-free* in `mlir::Operation::getBlock()`. The problem
originates from `CreateOperationRewrite::rollback()` removing blocks without
dropping the operations they contain, leaving dangling `UseList`s that are later
accessed by the verifier.

## Why this hits the described bug

During conversion, a clone of the `scf.for` op is created. When the converter
detects that types do not line up it rolls back, invoking the faulty
`CreateOperationRewrite::rollback()`. That function deletes the region’s block
list but *not* the ops inside; their `UseList`s still reference each other and
point back into already-freed memory, causing the crash.
