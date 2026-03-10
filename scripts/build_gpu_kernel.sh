#!/usr/bin/env bash

# Exit on error, undefined variable, or error in pipeline
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT="$ROOT_DIR/sky/elementwise_add_gpu.mlir"
OUT_DIR="/tmp/heir-gpu-add"
HEIR_OPT="$ROOT_DIR/bazel-bin/tools/heir-opt"
HEIR_TRANSLATE="$ROOT_DIR/bazel-bin/tools/heir-translate"
PTXAS="ptxas"
LLC="$(bazel info bazel-bin)/external/+_repo_rules+llvm-project/llvm/llc"
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

if [[ "${SKIP_CODEGEN:-0}" != "1" ]]; then
  for tool in "$LLC" "$PTXAS"; do
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

extract_kernel_module() {
  local input="$1"
  local output="$2"

  if ! sed -n '2p' "$input" | grep -q 'gpu\.module'; then
    echo "Expected a single lowered gpu.module in $input" >&2
    exit 1
  fi

  awk '
    {
      lines[++count] = $0
    }
    END {
      while (count > 0 && lines[count] ~ /^[[:space:]]*$/) {
        --count
      }
      if (count < 4) {
        exit 1
      }
      print "module {"
      for (i = 3; i <= count - 2; ++i) {
        print lines[i]
      }
      print "}"
    }
  ' "$input" > "$output"
}

"$HEIR_OPT" "$INPUT" -o "$HEIR_MLIR"
"$HEIR_OPT" "$HEIR_MLIR" \
  "--nvvm-attach-target=chip=${GPU_ARCH} O=3" \
  --convert-gpu-to-nvvm \
  --convert-scf-to-cf \
  --convert-cf-to-llvm \
  --convert-nvvm-to-llvm \
  --canonicalize \
  --cse \
  --reconcile-unrealized-casts \
  -o "$GPU_MLIR"
extract_kernel_module "$GPU_MLIR" "$KERNEL_MLIR"
"$HEIR_TRANSLATE" --mlir-to-llvmir "$KERNEL_MLIR" -o "$LLVM_IR"

if [[ "${SKIP_CODEGEN:-0}" == "1" ]]; then
  printf 'Wrote artifacts to %s\n' "$OUT_DIR"
  exit 0
fi

"$LLC" -march=nvptx64 -mcpu="$GPU_ARCH" "$LLVM_IR" -o "$PTX"
"$PTXAS" -arch="$GPU_ARCH" "$PTX" -o "$CUBIN"

printf 'Wrote artifacts to %s\n' "$OUT_DIR"
