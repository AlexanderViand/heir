#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT="$ROOT_DIR/sky/elementwise_add_linalg.mlir"
OUT_DIR="/tmp/heir-gpu-add"
HEIR_OPT="$ROOT_DIR/bazel-bin/tools/heir-opt"
HEIR_TRANSLATE="$ROOT_DIR/bazel-bin/tools/heir-translate"
PTXAS="ptxas"
CUOBJDUMP="cuobjdump"
NVCC="nvcc"
LLC="$ROOT_DIR/bazel-bin/external/+_repo_rules+llvm-project/llvm/llc"
RUNNER_SRC="$ROOT_DIR/sky/elementwise_add_gpu_smoketest.cpp"
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_ARCH="sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')"
else
  GPU_ARCH="sm_89"
fi

for tool in "$HEIR_OPT" "$HEIR_TRANSLATE"; do
  if ! command -v "$tool" >/dev/null 2>&1 && [[ ! -x "$tool" ]]; then
    echo "Required tool not found: $tool" >&2
    exit 1
  fi
done

if [[ ! -f "$INPUT" ]]; then
  echo "Input file not found: $INPUT" >&2
  exit 1
fi

if [[ ! -f "$RUNNER_SRC" ]]; then
  echo "Input file not found: $RUNNER_SRC" >&2
  exit 1
fi

if [[ "${SKIP_CODEGEN:-0}" != "1" ]]; then
  for tool in "$LLC" "$PTXAS" "$CUOBJDUMP" "$NVCC"; do
    if ! command -v "$tool" >/dev/null 2>&1 && [[ ! -x "$tool" ]]; then
      echo "Required tool not found: $tool" >&2
      exit 1
    fi
  done
fi

mkdir -p "$OUT_DIR"

HEIR_MLIR="$OUT_DIR/10-heir.mlir"
GPU_MLIR="$OUT_DIR/20-gpu.mlir"
KERNEL_MLIR="$OUT_DIR/30-kernel.mlir"
LLVM_IR="$OUT_DIR/40-kernel.ll"
PTX="$OUT_DIR/50-kernel.ptx"
CUBIN="$OUT_DIR/60-kernel.cubin"
SASS="$OUT_DIR/70-kernel.sass"
RUNNER="$OUT_DIR/80-smoketest"
RUN_LOG="$OUT_DIR/90-smoketest.log"

extract_kernel_module() {
  local input="$1"
  local output="$2"

  awk '
    /^[[:space:]]*gpu\.module / {
      print "module {"
      in_gpu_module = 1
      depth = 1
      next
    }

    !in_gpu_module {
      next
    }

    {
      if (depth == 1 && $0 ~ /^[[:space:]]*}[[:space:]]*$/) {
        print "}"
        exit 0
      }

      print $0

      line = $0
      opens = gsub(/\{/, "{", line)
      line = $0
      closes = gsub(/\}/, "}", line)
      depth += opens - closes
    }

    END {
      if (!in_gpu_module) {
        exit 1
      }
    }
  ' "$input" > "$output"
}

"$HEIR_OPT" "$INPUT" -o "$HEIR_MLIR"
echo "[gpu] lowered linalg input -> $HEIR_MLIR"
"$HEIR_OPT" "$HEIR_MLIR" \
  --convert-linalg-to-parallel-loops \
  --gpu-map-parallel-loops \
  --convert-parallel-loops-to-gpu \
  --gpu-kernel-outlining \
  --lower-affine \
  "--nvvm-attach-target=chip=${GPU_ARCH} O=3" \
  --convert-gpu-to-nvvm \
  --convert-scf-to-cf \
  --convert-cf-to-llvm \
  --convert-nvvm-to-llvm \
  --canonicalize \
  --cse \
  --reconcile-unrealized-casts \
  -o "$GPU_MLIR"
echo "[gpu] lowered gpu/nvvm module -> $GPU_MLIR"
extract_kernel_module "$GPU_MLIR" "$KERNEL_MLIR"
"$HEIR_TRANSLATE" --mlir-to-llvmir "$KERNEL_MLIR" -o "$LLVM_IR"
echo "[gpu] translated llvm ir -> $LLVM_IR"

if [[ "${SKIP_CODEGEN:-0}" == "1" ]]; then
  printf '[gpu] wrote artifacts to %s\n' "$OUT_DIR"
  exit 0
fi

"$LLC" -march=nvptx64 -mcpu="$GPU_ARCH" "$LLVM_IR" -o "$PTX"
"$PTXAS" -arch="$GPU_ARCH" "$PTX" -o "$CUBIN"
"$CUOBJDUMP" --dump-sass "$CUBIN" > "$SASS"
grep -q 'Function : elementwise_add_kernel' "$SASS"
"$NVCC" -std=c++17 -Wno-deprecated-gpu-targets "$RUNNER_SRC" -lcuda -o "$RUNNER"
echo "[gpu] assembled cubin -> $CUBIN"
echo "[gpu] disassembled sass -> $SASS"
echo "[gpu] smoke test log -> $RUN_LOG"
"$RUNNER" "$CUBIN" | tee "$RUN_LOG"
