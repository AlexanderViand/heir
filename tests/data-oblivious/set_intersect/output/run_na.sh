#!/bin/bash

# Run the mlir-to-openfhe-bgv pass
# bazel run //tools:heir-opt -- --mlir-to-openfhe-bgv='entry-function=set_intersect ciphertext-degree=16' $PWD/tests/data-oblivious/set_intersect/set_intersect.mlir > $PWD/tests/data-oblivious/set_intersect/output/openfhe_bgv.mlir

# Emit OpenFHE PKE header
# bazel run //tools:heir-translate -- --emit-openfhe-pke-header $PWD/tests/data-oblivious/set_intersect/output/openfhe_bgv.mlir > $PWD/tests/data-oblivious/set_intersect/output/heir_output.h

# Emit OpenFHE PKE cpp (implementation)
# bazel run //tools:heir-translate -- --emit-openfhe-pke $PWD/tests/data-oblivious/set_intersect/output/openfhe_bgv.mlir > $PWD/tests/data-oblivious/set_intersect/output/heir_output.cpp

# Run set_intersect test
# bazel run //tests/data-oblivious/set_intersect/output:set_intersect_main

# Lower basic MLIR to LLVM
bazel run //tools:heir-opt -- --heir-basic-mlir-to-llvm  $PWD/tests/data-oblivious/set_intersect/set_intersect.mlir > $PWD/tests/data-oblivious/set_intersect/output/set_intersect_llvm.mlir

# Build mlir-cpu-runner
bazel build @llvm-project//mlir:mlir-cpu-runner

# Run mlir-cpu-runner
bazel-bin/external/llvm-project/mlir/mlir-cpu-runner -e main -entry-point-result=struct $PWD/tests/data-oblivious/set_intersect/output/set_intersect_llvm.mlir

# TODO: Add mlir-translate to emit LLVM IR from MLIR
