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


# We want to have the function body above (numpy matmul) called in func.original
# and we want to emit `linalg.matmul` when running foo()/foo.eval()!
# Ideally, we'd also like to have uses of matmul outside of @compile() still
# correctly forward to np.matmul...

# @compile()
# def foo(
#     a: Secret[Tensor[32, 32, F32]], b: Secret[Tensor[32, 32, F32]]
# ) -> Secret[Tensor[32, 32, F32]]:
#   return matmul(a, b)
