"""Runtime e2e test for the OpenFHE Python emitter's multi-result/struct support.

The split-preprocessing pipeline emits a multi-result `mnist__preprocessing`
function (returning a generated `mnist__preprocessingStruct` of encoded weight
plaintexts) plus a `mnist__preprocessed` compute function. Calling these from
Python exercises the pybind bindings added for (a) the multi-result struct and
(b) the `Plaintext` element type -- without them, `mnist__preprocessing` cannot
return to Python ("Unregistered type") and its results cannot be unpacked.

This is intentionally the "split" path: weights are encoded once via
`mnist__preprocessing`, then fed into `mnist__preprocessed` per inference.
"""

import os
from typing import List
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import absl.testing.absltest
import tests.Examples.openfhe.ckks.preprocessing.mnist_openfhe_pybind as mnist

MODEL_PATH = "tests/Examples/common/mnist/data/traced_model.pt"
DATA_PATH = "tests/Examples/common/mnist/data"


class CustomMNISTTestDataset(Dataset):

  def __init__(
      self,
      data_root: str,
      normalize_mean: float = 0.1307,
      normalize_std: float = 0.3081,
  ):
    self.normalize_mean = normalize_mean
    self.normalize_std = normalize_std
    labels_path = os.path.join(data_root, "t10k-labels-idx1-ubyte")
    with open(labels_path, "rb") as f:
      f.read(8)
      labels = np.frombuffer(f.read(), dtype=np.uint8)
      self.targets = torch.tensor(labels, dtype=torch.long)
    images_path = os.path.join(data_root, "t10k-images-idx3-ubyte")
    with open(images_path, "rb") as f:
      f.read(16)
      images = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 1, 28, 28)
      self.data = torch.tensor(images, dtype=torch.float32) / 255.0

  def __len__(self) -> int:
    return len(self.data)

  def __getitem__(self, index: int):
    image = (self.data[index] - self.normalize_mean) / self.normalize_std
    return image, self.targets[index]


def load_weights(model_path: str) -> List[List[float]]:
  weights = []
  module = torch.jit.load(model_path)
  module.eval()
  for _, tensor in module.named_parameters():
    weights.append(tensor.cpu().contiguous().flatten().tolist())
  return weights


class MNISTSplitTest(absl.testing.absltest.TestCase):

  def test_split_preprocessing_roundtrip(self):
    weights = load_weights(MODEL_PATH)
    self.assertFalse(not weights, "load_weights failed")

    test_loader = DataLoader(
        CustomMNISTTestDataset(data_root=DATA_PATH), batch_size=1, shuffle=False
    )

    crypto_context = mnist.mnist__generate_crypto_context()
    key_pair = crypto_context.KeyGen()
    public_key = key_pair.publicKey
    secret_key = key_pair.secretKey
    crypto_context = mnist.mnist__configure_crypto_context(
        crypto_context, secret_key
    )

    # Encode the weight plaintexts ONCE. This call returns the generated
    # multi-result struct; reading its fields requires the Plaintext binding.
    pre = mnist.mnist__preprocessing(crypto_context, *weights[0:4])
    pts = []
    i = 0
    while hasattr(pre, f"arg{i}"):
      pts.append(getattr(pre, f"arg{i}"))
      i += 1
    self.assertGreater(i, 1, "expected a multi-field preprocessing struct")

    total = 4
    correct = 0
    samples_processed = 0
    for batch_data, batch_target in test_loader:
      if samples_processed >= total:
        break
      input_vector = batch_data.contiguous().flatten().tolist()
      input_encrypted = mnist.mnist__encrypt__arg4(
          crypto_context, input_vector, public_key
      )

      # Pass the pre-encoded plaintexts into the split-out compute function.
      output_encrypted = mnist.mnist__preprocessed(
          crypto_context, input_encrypted, *pts
      )

      output = mnist.mnist__decrypt__result0(
          crypto_context, output_encrypted, secret_key
      )
      label = batch_target.item()
      max_id = max(range(len(output)), key=lambda index: output[index])
      if max_id == label:
        correct += 1
      samples_processed += 1

    self.assertGreaterEqual(correct, 0.75 * total)


if __name__ == "__main__":
  absl.testing.absltest.main()
