#!/bin/bash

# Run the mlir-to-openfhe-bgv pass
# bazel run //tools:heir-opt -- --mlir-to-openfhe-bgv='entry-function=set_intersect_do ciphertext-degree=16' $PWD/tests/data-oblivious/set_intersect_do/set_intersect_do.mlir > $PWD/tests/data-oblivious/set_intersect/output/openfhe_bgv.mlir

# Emit OpenFHE PKE header
# bazel run //tools:heir-translate -- --emit-openfhe-pke-header $PWD/tests/data-oblivious/set_intersect_do/output/openfhe_bgv.mlir > $PWD/tests/data-oblivious/set_intersect/output/heir_output.h

# Emit OpenFHE PKE cpp (implementation)
# bazel run //tools:heir-translate -- --emit-openfhe-pke $PWD/tests/data-oblivious/set_intersect_do/output/openfhe_bgv.mlir > $PWD/tests/data-oblivious/set_intersect/output/heir_output.cpp

# Run set_intersect_do test
# bazel run //tests/data-oblivious/set_intersect/output:set_intersect_do_main

# Lower basic MLIR to LLVM
bazel run //tools:heir-opt -- --heir-basic-mlir-to-llvm  $PWD/tests/data-oblivious/set_intersect/set_intersect_do.mlir > $PWD/tests/data-oblivious/set_intersect/output/set_intersect_do_llvm.mlir

# Build mlir-cpu-runner
bazel build @llvm-project//mlir:mlir-cpu-runner

# Run mlir-cpu-runner and measure execution time
time $PWD/bazel-bin/external/llvm-project/mlir/mlir-cpu-runner -e main -entry-point-result=void --shared-libs="bazel-bin/external/llvm-project/mlir/libmlir_c_runner_utils.so,bazel-bin/external/llvm-project/mlir/libmlir_runner_utils.so" $PWD/tests/data-oblivious/set_intersect/output/set_intersect_do_llvm.mlir

# TODO: Use mlir-translate to emit LLVM IR from MLIR
