from heir.mlir import Secret, Tensor
from numpy.typing import NDArray
import numpy as np


# FIXME: Currently a no-op
# TODO: Augment the implementation function with a check
#  to ensure it's not being called with a Ciphertext / Secret /whatever
def mlir_op(name: str):
  def decorator(func):
    func.__name__ = name
    return func

  return decorator


@mlir_op("linalg.matmul")
def matmul(
    lhs: Secret[Tensor] | Tensor | NDArray,
    rhs: Secret[Tensor] | Tensor | NDArray,
) -> Tensor:
  return np.matmul(lhs, rhs)  # type: ignore
