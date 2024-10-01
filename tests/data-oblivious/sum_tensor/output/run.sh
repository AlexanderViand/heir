#!/bin/bash
bazel run //tools:heir-opt -- --mlir-to-openfhe-bgv='entry-function=sum_tensor ciphertext-degree=32' $PWD/tests/data-oblivious/sum_tensor/sum_tensor.mlir > $PWD/tests/data-oblivious/sum_tensor/output/openfhe_bgv.mlir
bazel run //tools:heir-translate -- --emit-openfhe-pke-header $PWD/tests/data-oblivious/sum_tensor/output/openfhe_bgv.mlir > $PWD/tests/data-oblivious/sum_tensor/output/heir_output.h
bazel run //tools:heir-translate -- --emit-openfhe-pke $PWD/tests/data-oblivious/sum_tensor/output/openfhe_bgv.mlir > $PWD/tests/data-oblivious/sum_tensor/output/heir_output.cpp
bazel run //tests/data-oblivious/sum_tensor/output:sum_tensor_main
