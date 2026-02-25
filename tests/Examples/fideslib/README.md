# FIDESlib Smoke Test

This target validates a minimal FIDESlib flow from HEIR:

- context/key generation
- encrypt
- `EvalAdd` / `EvalMult`
- decrypt and value checks

Run:

```bash
bazel test //tests/Examples/fideslib:smoke_test --//:enable_fideslib=1 --test_output=all
```

## Local FIDESlib wiring

For local iteration, HEIR is currently wired to a sibling checkout via:

- `heir/bazel/extensions.bzl`:
  - `local_repository(name = "fideslib", path = "../FIDESlib")`

## Required host dependencies (Ubuntu)

```bash
sudo apt-get install -y g++-12 libstdc++-12-dev libomp-dev lld libgmp-dev libntl-dev
```

Also required:

- CUDA toolkit (tested with CUDA 12.x)
