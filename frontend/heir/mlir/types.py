"""Defines Python type annotations for MLIR types."""

from abc import ABC, abstractmethod
from typing import Tuple, Generic, Self, TypeVar, TypeVarTuple, get_args, get_origin
from numba.core.types import Type as NumbaType
from numba.core.types import boolean, int8, int16, int32, int64, float32, float64

T = TypeVar("T")
Ts = TypeVarTuple("Ts")


operator_error_message = (
    "MLIRTypeAnnotation should only be used for annotations, never at runtime."
)


class MLIRTypeAnnotation(ABC):

  @staticmethod
  @abstractmethod
  def numba_type() -> NumbaType:
    raise NotImplementedError(
        "No numba type exists for a generic MLIRTypeAnnotation"
    )

  # FIXME: Instead of defining these here, only define them where they make sense for the type
  # FIXME: For Secret, see if we can make them available depending on the wrapped type having the operator
  # FIXME: Oh, but probably not statically/in a way that pylance/pyright will understand :(
  # FIXME: or is there some overload-style thing to signal stuff like this to type checkers?
  def __add__(self, other) -> Self:
    raise RuntimeError(operator_error_message)

  def __sub__(self, other) -> Self:
    raise RuntimeError(operator_error_message)

  def __mul__(self, other) -> Self:
    raise RuntimeError(operator_error_message)


class Secret(Generic[T], MLIRTypeAnnotation):

  @staticmethod
  def numba_type() -> NumbaType:
    raise NotImplementedError("No numba type exists for a generic Secret")


class Tensor(Generic[*Ts], MLIRTypeAnnotation):

  @staticmethod
  def numba_type() -> NumbaType:
    raise NotImplementedError("No numba type exists for a generic Tensor")


class F32(MLIRTypeAnnotation):
  # TODO (#1162): For CKKS/Float: allow specifying actual intended precision/scale and warn/error if not achievable  @staticmethod
  @staticmethod
  def numba_type() -> NumbaType:
    return float32


class F64(MLIRTypeAnnotation):
  # TODO (#1162): For CKKS/Float: allow specifying actual intended precision/scale and warn/error if not achievable  @staticmethod
  @staticmethod
  def numba_type() -> NumbaType:
    return float64


class I1(MLIRTypeAnnotation):

  @staticmethod
  def numba_type() -> NumbaType:
    return boolean


class I8(MLIRTypeAnnotation):

  @staticmethod
  def numba_type() -> NumbaType:
    return int8


class I16(MLIRTypeAnnotation):

  @staticmethod
  def numba_type() -> NumbaType:
    return int16


class I32(MLIRTypeAnnotation):

  @staticmethod
  def numba_type() -> NumbaType:
    return int32


class I64(MLIRTypeAnnotation):

  @staticmethod
  def numba_type() -> NumbaType:
    return int64


# Helper functions


def to_numba_type(type: type) -> Tuple[NumbaType, bool]:
  if get_origin(type) == Secret:
    if len(get_args(type)) != 1:
      raise TypeError(
          f"Secret should contain a single type argument, but found {type}"
      )
    numba_type, is_nested_secret = to_numba_type(get_args(type)[0])
    if is_nested_secret:
      raise TypeError("Nested Secrets are not yet supported.")
    return numba_type, True

  if get_origin(type) == Tensor:
    args = get_args(type)
    if len(args) < 2:
      raise TypeError(
          "Tensor should contain a type and at least one dimension, but found"
          f" {type}"
      )
    inner_type = args[-1]
    if get_origin(inner_type) == Tensor:
      raise TypeError("Nested Tensors are not yet supported.")
    ty, is_secret = to_numba_type(inner_type)
    # This is slightly cursed, as numba constructs array types via slice syntax
    # Cf. https://numba.pydata.org/numba-doc/dev/reference/types.html#arrays
    ty = ty[(slice(None),) * (len(args) - 1)]
    # We augment the type object with `shape` for the actual sizes
    ty.shape = args[:-1]  # type: ignore
    return ty, is_secret

  if issubclass(type, MLIRTypeAnnotation):
    return type.numba_type(), False

  raise TypeError(f"Unsupported type annotation: {type}, {get_origin(type)}")


def parse_annotations(
    annotations,
) -> tuple[list[NumbaType], list[int], NumbaType | None]:
  """Converts a python type annotation to a list of numba types.
  Args:
    annotations: A dictionary of type annotations, e.g. func.__annotations__
  Returns:
    A tuple of (args, secret_args, rettype) where:
    - args: a list of numba types for the function arguments
    - secret_args: a list of indices of secret arguments
    - rettype: the numba type of the return value
  """
  if not annotations:
    raise TypeError("Function is missing type annotations.")
  args: list[NumbaType] = []
  secret_args: list[int] = []
  rettype = None
  for idx, (name, arg_type) in enumerate(annotations.items()):
    if name == "return":
      # TODO (#1162): if user has annotated return type as secret,
      # annotate it as such on the function (and make HEIR check it!)
      rettype, _ = to_numba_type(arg_type)
      continue
    else:
      numba_type, is_secret = to_numba_type(arg_type)
      args.append(numba_type)
      if is_secret:
        secret_args.append(idx)
  return args, secret_args, rettype
