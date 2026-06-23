"""Extract + BatchNorm-fold the Orion MLP weights into a flat f32 blob.

The Orion MLP is fc1 -> bn1 -> Quad -> fc2 -> bn2 -> Quad -> fc3 (Orion's
trained MNIST checkpoint). We fold each eval-mode
BatchNorm1d into its preceding Linear -- the SAME fold the linalg export does
(see the comment in orion_mlp.mlir) -- so the C++ e2e harness sees a plain
fc -> quad -> fc -> quad -> fc MLP whose weights match the lowered IR.

  BN(y) = scale * y + shift,  scale = gamma / sqrt(var + eps), shift = beta - mean*scale
  folded into y = W x + b:     W' = scale[:,None] * W,  b' = scale * b + shift

Used by a build-time genrule so the C++ harness (which can't read a .pth) gets
the trained, folded weights without committing any derived data. Images,
labels and the plaintext reference are handled directly in the C++ test.

Usage: dump_weights.py <mlp.pth> <out.bin>
Layout (little-endian f32): fc1.w[128,784], fc1.b[128], fc2.w[128,128],
                            fc2.b[128], fc3.w[10,128], fc3.b[10].
Uses only torch + struct (the build-time torch env has no numpy).
"""

import struct
import sys

import torch

EPS = 1e-5  # orion.nn.BatchNorm1d default


def _fold(w, b, bn_w, bn_b, mean, var):
  scale = bn_w / torch.sqrt(var + EPS)
  shift = bn_b - mean * scale
  return w * scale.unsqueeze(1), b * scale + shift


def main(model_path: str, out_path: str) -> None:
  sd = torch.load(model_path, map_location="cpu", weights_only=False)
  if isinstance(sd, dict) and "state_dict" in sd:
    sd = sd["state_dict"]
  w1, b1 = _fold(
      sd["fc1.weight"],
      sd["fc1.bias"],
      sd["bn1.weight"],
      sd["bn1.bias"],
      sd["bn1.running_mean"],
      sd["bn1.running_var"],
  )
  w2, b2 = _fold(
      sd["fc2.weight"],
      sd["fc2.bias"],
      sd["bn2.weight"],
      sd["bn2.bias"],
      sd["bn2.running_mean"],
      sd["bn2.running_var"],
  )
  w3, b3 = sd["fc3.weight"], sd["fc3.bias"]

  tensors = [w1, b1, w2, b2, w3, b3]
  assert [list(t.shape) for t in tensors] == [
      [128, 784],
      [128],
      [128, 128],
      [128],
      [10, 128],
      [10],
  ], [list(t.shape) for t in tensors]
  with open(out_path, "wb") as f:
    for t in tensors:
      vals = t.detach().cpu().contiguous().float().flatten().tolist()
      f.write(struct.pack("<%df" % len(vals), *vals))


if __name__ == "__main__":
  main(sys.argv[1], sys.argv[2])
