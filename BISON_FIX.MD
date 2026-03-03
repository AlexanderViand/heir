# Bison fix notes

## Problem

Builds fail in bison/gnulib with a conflict on
`posix_spawn_file_actions_addchdir`.

Typical error:

```text
.../external/bison+/lib/spawn.h:... conflicting types for 'posix_spawn_file_actions_addchdir'
.../MacOSX.sdk/usr/include/spawn.h:... previous declaration is here
```

One target that hits it: `//tests:test_utilities`.

## Environment where this was reproduced

- macOS `26.3` (`25D125`)
- Darwin `25.3.0` arm64
- Xcode `26.2` (`17C52`)
- SDK `MacOSX26.2.sdk`
- Apple clang `17.0.0`

## What is not the issue

- Xcode install is fine.
- `xcode-select` points to full Xcode.
- Bazel sees SDK include paths.

So this is not a broken local Xcode setup.

## Root cause

The BCR `bison` packaging has stale macOS config values.

In `config_macos.bzl`, `HAVE_POSIX_SPAWN_FILE_ACTIONS_ADDCHDIR` is set to `0`.
On this SDK, `<spawn.h>` already declares it. So gnulib generates a fallback
declaration/implementation that now collides.

## Proper fix

Do this in repo, not in Bazel cache:

1. Add a `MODULE.bazel` override for `bison`.
1. Add a patch file under `patches/`.
1. Patch bison/gnulib generation so fallback code is only used when the symbol
   is actually missing.

## Guard logic (important)

Do **not** use hardcoded SDK threshold checks (like `>= 260000`) as the main
fix. That is brittle.

Use declaration/config checks:

```c
#if !HAVE_POSIX_SPAWN_FILE_ACTIONS_ADDCHDIR
/* fallback declaration/implementation */
#endif
```

If available in this path, raw declaration checks are even better:

```c
#if !HAVE_RAW_DECL_POSIX_SPAWN_FILE_ACTIONS_ADDCHDIR
/* fallback declaration/implementation */
#endif
```

This works for both:

- newer SDKs that already have the symbol,
- older SDKs that do not.

## Upstream

Open/fix in:

1. Bazel Central Registry bison module packaging (primary).
1. bison/gnulib upstream if needed (secondary).

## Verify after patch

```bash
bazel build '@@bison+//:gnulib'
bazel build //tests:test_utilities
```
