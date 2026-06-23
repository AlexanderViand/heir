"""Bake CHEDDAR's exact canonical per-level scales into a cheddar-level .mlir.

The cheddar-to-emitc emitter emits each `cheddar.encode`'s `scale` attribute
verbatim, but `--scheme-to-cheddar` only records a NOMINAL scale (a clean 2^k).
CHEDDAR's actual per-level scale is prime-derived and drifts from 2^k (rescale
divides by the exact prime product), and CHEDDAR rejects mismatches beyond
1e-12. So this pre-pass rewrites each encode's `scale` to the exact value the
emitter used to materialise at runtime as `ctx->param_.GetScale(level)^k`
(k = log2(nominal) / logDefaultScale, i.e. 1 for a fresh plaintext, 2 for a
doubled post-mult one). Any `cheddar.exact_scale` override (e.g. the relu biases
whose ciphertext drifted off canonical GetScale^k) is folded into `scale` and
dropped.

This is the manual stand-in until precise scale management computes the exact
scales in the higher-level pipeline. The GetScale(level) table is parameter-set
specific; obtain it by constructing a `cheddar::Parameter<word>` with the test's
params and printing `param.GetScale(level)` (%.17g) for each level.

Usage:
  bake_scales.py <mlir> <logDefaultScale> <level>:<getscale> [<level>:<getscale> ...]
"""

import math
import re
import struct
import sys

ENCODE = "cheddar.encode "  # trailing space excludes cheddar.encode_constant
# MLIR prints f64 attrs as either a hex bit-pattern (0x....) or a decimal float.
FLOAT = r"(0x[0-9A-Fa-f]+|[0-9.eE+-]+)"
SCALE_RE = re.compile(r"(?<![.\w])scale = " + FLOAT + r" : f64")
LEVEL_RE = re.compile(r"level = (\d+) : i64")
EXACT_RE = re.compile(r"\s*,\s*cheddar\.exact_scale = " + FLOAT + r" : f64")


def parse_f64(s: str) -> float:
  if s.startswith("0x"):
    return struct.unpack("<d", struct.pack("<Q", int(s, 16)))[0]
  return float(s)


def hex_f64(x: float) -> str:
  return "0x%016X" % struct.unpack("<Q", struct.pack("<d", x))[0]


def bake_line(line, getscale, log_default):
  if ENCODE not in line:
    return line, 0
  mlvl, mscale = LEVEL_RE.search(line), SCALE_RE.search(line)
  if not mlvl or not mscale:
    return line, 0
  level, nominal = int(mlvl.group(1)), parse_f64(mscale.group(1))
  mexact = EXACT_RE.search(line)
  if mexact:
    baked = parse_f64(mexact.group(1))
    line = EXACT_RE.sub("", line, count=1)
  else:
    log_scale = round(math.log2(nominal))
    k = (log_scale // log_default
         if log_scale >= log_default and log_scale % log_default == 0 else 1)
    baked = getscale[level] ** k
  line = SCALE_RE.sub("scale = " + hex_f64(baked) + " : f64", line, count=1)
  return line, 1


def main() -> None:
  path, log_default = sys.argv[1], int(sys.argv[2])
  getscale = {int(kv.split(":")[0]): float(kv.split(":")[1]) for kv in sys.argv[3:]}
  out, n = [], 0
  for line in open(path).read().splitlines(keepends=True):
    line, c = bake_line(line, getscale, log_default)
    n += c
    out.append(line)
  with open(path, "w") as f:
    f.write("".join(out))
  print(f"baked {n} cheddar.encode scales in {path}")


if __name__ == "__main__":
  main()
