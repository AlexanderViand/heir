#!/bin/bash
# Run secretize and wrap-generic passes
# bazel run //tools:heir-opt -- --secretize=entry-function="simple_if" --wrap-generic --convert-if-to-select $PWD/tests/data-oblivious/simple_if/simple_if.mlir >  $PWD/tests/data-oblivious/simple_if/simple_if_output.mlir

# Run the mlir-to-openfhe-bgv pass
# bazel run //tools:heir-opt -- --mlir-to-openfhe-bgv='entry-function=simple_if ciphertext-degree=8' $PWD/tests/data-oblivious/simple_if/simple_if.mlir > $PWD/tests/data-oblivious/simple_if/output/openfhe_bgv.mlir

# Emit OpenFHE PKE header
# bazel run //tools:heir-translate -- --emit-openfhe-pke-header $PWD/tests/data-oblivious/simple_if/output/openfhe_bgv.mlir > $PWD/tests/data-oblivious/simple_if/output/heir_output.h

# Emit OpenFHE PKE cpp (implementation)
# bazel run //tools:heir-translate -- --emit-openfhe-pke $PWD/tests/data-oblivious/simple_if/output/openfhe_bgv.mlir > $PWD/tests/data-oblivious/simple_if/output/heir_output.cpp

# Run simple_if test
# bazel run //tests/data-oblivious/simple_if/output:simple_if_main

# Lower basic MLIR to LLVM
# bazel run //tools:heir-opt -- --heir-basic-mlir-to-llvm  $PWD/tests/data-oblivious/simple_if/simple_if.mlir > $PWD/tests/data-oblivious/simple_if/output/simple_if_llvm.mlir

# Build mlir-cpu-runner
bazel build @llvm-project//mlir:mlir-cpu-runner

# Run mlir-cpu-runner
bazel-bin/external/llvm-project/mlir/mlir-cpu-runner -e main -entry-point-result=i32 $PWD/tests/data-oblivious/simple_if/output/simple_if_llvm.mlir

# TODO: Add mlir-translate to emit LLVM IR from MLIR
