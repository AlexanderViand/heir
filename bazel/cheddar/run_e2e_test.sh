#!/bin/bash
# Launcher for the CHEDDAR GPU e2e tests (see e2e.bzl): points the
# system-toolchain-compiled test binary at libcheddar.so and the CUDA
# runtime, then execs it.
#
# $1 = test binary (runfiles-relative), $2 = @cheddar install dir.
set -eu
export LD_LIBRARY_PATH="$2/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$1"
