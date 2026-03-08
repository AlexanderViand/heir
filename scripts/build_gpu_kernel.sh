#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/build_gpu_kernel.sh INPUT.mlir [OUT_DIR]

Builds one GPU kernel artifact bundle:
  1. HEIR lowering with bazel-bin/tools/heir-opt
  2. GPU/NVVM lowering with bazel-bin/tools/heir-opt
  3. LLVM IR export with bazel-bin/tools/heir-translate
  4. PTX generation with llc
  5. cubin assembly with ptxas

Outputs:
  10-heir.mlir
  20-gpu.mlir
  30-kernel.ll
  40-kernel.ptx
  50-kernel.cubin

Environment:
  HEIR_OPT        Override bazel-bin/tools/heir-opt
  HEIR_TRANSLATE  Override bazel-bin/tools/heir-translate
  LLC             Override llc
  PTXAS           Override ptxas
  HEIR_PASSES     Extra flags for the first heir-opt invocation
  GPU_PASSES      Extra flags for the second heir-opt invocation
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage >&2
  exit 1
fi

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  usage
  exit 0
fi

INPUT="$1"
if [[ ! -f "$INPUT" ]]; then
  echo "Input file not found: $INPUT" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${2:-$(mktemp -d "${TMPDIR:-/tmp}/heir-gpu.XXXXXX")}"

HEIR_OPT="${HEIR_OPT:-$ROOT_DIR/bazel-bin/tools/heir-opt}"
HEIR_TRANSLATE="${HEIR_TRANSLATE:-$ROOT_DIR/bazel-bin/tools/heir-translate}"
LLC="${LLC:-llc}"
PTXAS="${PTXAS:-ptxas}"

for tool in "$HEIR_OPT" "$HEIR_TRANSLATE" "$LLC" "$PTXAS" nvidia-smi; do
  if ! command -v "$tool" >/dev/null 2>&1 && [[ ! -x "$tool" ]]; then
    echo "Required tool not found: $tool" >&2
    exit 1
  fi
done

mkdir -p "$OUT_DIR"

HEIR_MLIR="$OUT_DIR/10-heir.mlir"
GPU_MLIR="$OUT_DIR/20-gpu.mlir"
LLVM_IR="$OUT_DIR/30-kernel.ll"
PTX="$OUT_DIR/40-kernel.ptx"
CUBIN="$OUT_DIR/50-kernel.cubin"

GPU_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')"
if [[ -z "$GPU_ARCH" ]]; then
  echo "Failed to detect GPU compute capability with nvidia-smi." >&2
  exit 1
fi

read -r -a HEIR_PASS_ARGS <<< "${HEIR_PASSES:-}"
read -r -a GPU_PASS_ARGS <<< "${GPU_PASSES:-}"

"$HEIR_OPT" "$INPUT" "${HEIR_PASS_ARGS[@]}" -o "$HEIR_MLIR"
"$HEIR_OPT" "$HEIR_MLIR" "${GPU_PASS_ARGS[@]}" -o "$GPU_MLIR"
"$HEIR_TRANSLATE" --mlir-to-llvmir "$GPU_MLIR" -o "$LLVM_IR"
"$LLC" -march=nvptx64 -mcpu="sm_${GPU_ARCH}" "$LLVM_IR" -o "$PTX"
"$PTXAS" -arch="sm_${GPU_ARCH}" "$PTX" -o "$CUBIN"

printf 'Wrote artifacts to %s\n' "$OUT_DIR"
