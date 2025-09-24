import os
import time
from typing import List
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import absl.testing.absltest
import tests.Examples.openfhe.ckks.mnist.mnist_openfhe_pybind as mnist

MODEL_PATH = "tests/Examples/openfhe/ckks/mnist/data/traced_model.pt"
DATA_PATH = "tests/Examples/openfhe/ckks/mnist/data"


class CustomMNISTTestDataset(Dataset):
  """This custom dataset loads the raw MNIST test data and labels

  from the files specified by `data_root`.
  It applies the Normalize transform manually during loading.
  """

  def __init__(
      self,
      data_root: str,
      normalize_mean: float = 0.1307,
      normalize_std: float = 0.3081,
  ):
    self.data_root = data_root
    self.normalize_mean = normalize_mean
    self.normalize_std = normalize_std

    labels_path = os.path.join(self.data_root, "t10k-labels-idx1-ubyte")
    with open(labels_path, "rb") as f:
      # Skip header
      f.read(8)
      labels = np.frombuffer(f.read(), dtype=np.uint8)
      self.targets = torch.tensor(labels, dtype=torch.long)

    images_path = os.path.join(self.data_root, "t10k-images-idx3-ubyte")
    with open(images_path, "rb") as f:
      # Skip header
      f.read(16)
      images = np.frombuffer(f.read(), dtype=np.uint8)
      images = images.reshape(-1, 1, 28, 28)
      self.data = torch.tensor(images, dtype=torch.float32) / 255.0

  def __len__(self) -> int:
    return len(self.data)

  def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    image = self.data[index]
    label = self.targets[index]

    # Apply normalization manually (C++ .map(transforms::Normalize))
    # (X - mean) / std_dev
    image = (image - self.normalize_mean) / self.normalize_std

    return image, label


def load_weights(model_path: str) -> List[List[float]]:
  """Loads JIT-traced model parameters and converts them into a list of lists of floats."""
  weights = []
  module = torch.jit.load(model_path)
  module.eval()
  print(f"Successfully loaded {model_path}")
  print("Model parameters:")
  for name, tensor in module.named_parameters():
    print(f"  {name}: {list(tensor.size())}")
    # Conversion: .contiguous().flatten().tolist()
    tensor_data = tensor.cpu().contiguous().flatten().tolist()
    weights.append(tensor_data)
  return weights


class MNISTTest(absl.testing.absltest.TestCase):

  def test_run_test(self):
    weights = load_weights(MODEL_PATH)
    self.assertFalse(not weights, "load_weights failed")

    # Note: The C++ code this was ported from uses a custom stack transform
    # after normalization. In Python, DataLoader handles batching, which
    # effectively stacks the tensors.
    test_dataset = CustomMNISTTestDataset(data_root=DATA_PATH)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,  # SequentialSampler equivalent
    )

    crypto_context = mnist.mnist__generate_crypto_context()
    key_pair = crypto_context.KeyGen()
    public_key = key_pair.publicKey
    secret_key = key_pair.secretKey
    crypto_context = mnist.mnist__configure_crypto_context(
        crypto_context, secret_key
    )

    # 4. Evaluation Loop
    total = 4
    correct = 0
    samples_processed = 0

    for batch_data, batch_target in test_loader:
      if samples_processed >= total:
        break

      input_tensor = batch_data.contiguous()  # (1, 1, 28, 28)
      input_vector = input_tensor.flatten().tolist()
      input_encrypted = mnist.mnist__encrypt__arg4(
          crypto_context, input_vector, public_key
      )

      start_time = time.time()
      output_encrypted = mnist.mnist(
          crypto_context, *weights[0:4], input_encrypted
      )
      end_time = time.time()

      time_elapsed_ms = (end_time - start_time) * 1000.0
      print(f"CPU time used: {time_elapsed_ms:.2f} ms")

      output = [0.0] * 10
      output = mnist.mnist__decrypt__result0(
          crypto_context, output_encrypted, secret_key
      )
      label = batch_target.item()
      max_id = max(range(len(output)), key=lambda index: output[index])

      # NOTE: For the test to pass with the default placeholder 'output',
      # the `mnist__decrypt__result0` function must be fixed to return an
      # output vector that results in `max_id == label` for 90% of samples.
      if max_id == label:
        correct += 1

      print(f"max_id: {max_id}, label: {label}")
      samples_processed += 1

    self.assertGreaterEqual(correct, 0.75 * total)


if __name__ == "__main__":
  absl.testing.absltest.main()
