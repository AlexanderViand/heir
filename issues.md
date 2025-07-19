# Significant issues discovered during code inspection

The following problems may lead to compiler crashes, memory‐safety errors, or
silent wrong-code generation. Cosmetic defects (typos, formatting, missing docs,
etc.) are intentionally ignored.

1. **`lib/Utils/ContextAwareDialectConversion.cpp`**

   - *Tag overflow*: `UnresolvedMaterializationRewrite` uses
     `llvm::PointerIntPair<const ContextAwareTypeConverter *, 2, MaterializationKind>`.
     If `MaterializationKind` ever gains a third value the two tag bits
     overflow, resulting in undefined behaviour when the pair is read.

   - *Incomplete rollback in `CreateOperationRewrite`*: `rollback()` removes all
     blocks from the operation’s regions but never destroys the operations
     inside those blocks before calling `op->erase()`. This leaks IR objects and
     leaves dangling uses in debug builds.

   - *Dangling mapping entries in `UnresolvedMaterializationRewrite::rollback`*:
     the method erases the `op` only **after** removing it from the
     `unresolvedMaterializations` map. Other entries in `mapping` may still
     reference the soon-to-be-destroyed values, leading to use-after-free when
     they are accessed later.

1. **`lib/Analysis/SelectVariableNames/SelectVariableNames.cpp`**

   - *Lifetime bug*: `suggestNameForValue()` returns `defaultPrefix` **by
     value**. `assignName` keeps a reference to that returned string; when the
     default-prefix path is taken the reference points to a destroyed temporary
     -> immediate UB.

1. **`lib/Target/OpenFhePke/OpenFhePkeEmitter.cpp`**

   - *Duplicate variable declarations*: depending on the order in which a
     `Value` is printed, `emitTypedAssignPrefix` may output both a
     `const auto& v = …;` and later a typed declaration such as
     `int64_t v = …;`, causing the generated C++ to fail to compile.

   - *Fragile overload resolution*: the float branch of
     `printOperation(arith::ConstantOp)` relies on implicit conversions that
     will break once the upstream MLIR API changes (already proposed).

1. **`lib/Pipelines/ArithmeticPipelineRegistration.cpp`**

   - *Double loop-unrolling*: when the `disableLoopUnroll` option is false the
     pipeline unrolls loops once explicitly and again inside the affine
     sub-pipeline. Large kernels hit MLIR’s pass-limit assert and crash.

1. **`lib/Transforms/SecretInsertMgmt/*`**

   - *Dangling uses*: several patterns erase a `tensor.extract` without first
     replacing all of its uses. If the extract had multiple uses, later
     verifiers will crash due to dangling def-use chains.

1. **Global pattern**

   - Many passes keep long-lived `llvm::DenseMap<Value, …>` caches while also
     performing RAUW mutations. Neither `DenseMap` nor `Value` update those
     keys, so stale pointers remain and may be dereferenced later →
     non-deterministic crashes.
