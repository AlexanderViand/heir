#!/bin/bash
bazel run //tools:heir-opt -- --mlir-to-openfhe-bgv='entry-function=set_intersect ciphertext-degree=16' $PWD/tests/data-oblivious/set_intersect/set_intersect.mlir > $PWD/tests/data-oblivious/set_intersect/output/openfhe_bgv.mlir
# bazel run //tools:heir-translate -- --emit-openfhe-pke-header $PWD/tests/data-oblivious/set_intersect/output/openfhe_bgv.mlir > $PWD/tests/data-oblivious/set_intersect/output/heir_output.h
# bazel run //tools:heir-translate -- --emit-openfhe-pke $PWD/tests/data-oblivious/set_intersect/output/openfhe_bgv.mlir > $PWD/tests/data-oblivious/set_intersect/output/heir_output.cpp
# bazel run //tests/data-oblivious/set_intersect/output:set_intersect_main
