#!/bin/bash
# bazel run //tools:heir-opt -- --secretize=entry-function="simple_if" --wrap-generic --convert-if-to-select $PWD/tests/data-oblivious/simple_if/simple_if.mlir >  $PWD/tests/data-oblivious/simple_if/simple_if_output.mlir
bazel run //tools:heir-opt -- --mlir-to-openfhe-bgv='entry-function=simple_if ciphertext-degree=8' $PWD/tests/data-oblivious/simple_if/simple_if.mlir > $PWD/tests/data-oblivious/simple_if/output/openfhe_bgv.mlir
# bazel run //tools:heir-translate -- --emit-openfhe-pke-header $PWD/tests/data-oblivious/simple_if/output/openfhe_bgv.mlir > $PWD/tests/data-oblivious/simple_if/output/heir_output.h
# bazel run //tools:heir-translate -- --emit-openfhe-pke $PWD/tests/data-oblivious/simple_if/output/openfhe_bgv.mlir > $PWD/tests/data-oblivious/simple_if/output/heir_output.cpp
# bazel run //tests/data-oblivious/simple_if/output:simple_if_main
