"""Extract the MNIST MLP weights from the JIT-traced model into a flat f32 blob.

Used by a build-time genrule so the C++ e2e harness (which can't read a .pt)
gets the trained weights without committing any derived data. The images,
labels and plaintext reference are handled directly in the C++ test.

Usage: dump_weights.py <traced_model.pt> <out.bin>
Layout (little-endian f32): model.0.weight[512,784], model.0.bias[512],
                            model.2.weight[10,512], model.2.bias[10].

Uses only torch + struct (the build-time torch env has no numpy).
"""

import struct
import sys

import torch


def main(model_path: str, out_path: str) -> None:
  module = torch.jit.load(model_path)
  module.eval()
  tensors = [
      t.detach().cpu().contiguous().float().flatten()
      for _, t in module.named_parameters()
  ]
  sizes = [list(t.size()) for _, t in module.named_parameters()]
  assert sizes == [[512, 784], [512], [10, 512], [10]], sizes
  with open(out_path, "wb") as f:
    for t in tensors:
      vals = t.tolist()
      f.write(struct.pack("<%df" % len(vals), *vals))


if __name__ == "__main__":
  main(sys.argv[1], sys.argv[2])
