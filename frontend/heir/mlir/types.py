"""Defines Python type annotations for MLIR types."""

from abc import ABC, abstractmethod
from typing import Generic, Self, TypeVar, TypeVarTuple, get_args, get_origin
from numba.core.types import Type as NumbaType
from numba.core.types import boolean, int8, int16, int32, int64, float32, float64

T = TypeVar("T")
Ts = TypeVarTuple("Ts")

operator_error_message = "MLIRType should only be used for annotations."


class MLIRType(ABC):

  @staticmethod
  @abstractmethod
  def numba_type() -> NumbaType:
    raise NotImplementedError("No numba type exists for a generic MLIRType")

  def __add__(self, other) -> Self:
    raise RuntimeError(operator_error_message)

  def __sub__(self, other) -> Self:
    raise RuntimeError(operator_error_message)

  def __mul__(self, other) -> Self:
    raise RuntimeError(operator_error_message)


class Secret(Generic[T], MLIRType):

  @staticmethod
  def numba_type() -> NumbaType:
    raise NotImplementedError("No numba type exists for a generic Secret")


class Tensor(Generic[*Ts], MLIRType):

  @staticmethod
  def numba_type() -> NumbaType:
    raise NotImplementedError("No numba type exists for a generic Tensor")


class F32(MLIRType):
  # TODO (#1162): For CKKS/Float: allow specifying actual intended precision/scale and warn/error if not achievable  @staticmethod
  @staticmethod
  def numba_type() -> NumbaType:
    return float32


class F64(MLIRType):
  # TODO (#1162): For CKKS/Float: allow specifying actual intended precision/scale and warn/error if not achievable  @staticmethod
  @staticmethod
  def numba_type() -> NumbaType:
    return float64


class I1(MLIRType):

  @staticmethod
  def numba_type() -> NumbaType:
    return boolean


class I8(MLIRType):

  @staticmethod
  def numba_type() -> NumbaType:
    return int8


class I16(MLIRType):

  @staticmethod
  def numba_type() -> NumbaType:
    return int16


class I32(MLIRType):

  @staticmethod
  def numba_type() -> NumbaType:
    return int32


class I64(MLIRType):

  @staticmethod
  def numba_type() -> NumbaType:
    return int64


# Helper functions


def to_numba_type(type: type) -> NumbaType:
  if get_origin(type) == Secret:
    raise TypeError(
        "Secret type should not appear inside another type annotation."
    )

  if get_origin(type) == Tensor:
    args = get_args(type)
    if len(args) != 2:
      raise TypeError(
          "Tensor should contain exactly two elements: a shape list and a"
          f" type, but found {type}"
      )
    shape = args[0]
    inner_type = args[1]
    if get_origin(inner_type) == Tensor:
      raise TypeError("Nested Tensors are not yet supported.")
    # This is slightly cursed, as numba constructs array types via slice syntax
    # Cf. https://numba.pydata.org/numba-doc/dev/reference/types.html#arrays
    ty = to_numba_type(inner_type)[(slice(None),) * len(shape)]
    # We augment the type object with `shape` for the actual sizes
    ty.shape = shape  # type: ignore
    return ty

  if issubclass(type, MLIRType):
    return type.numba_type()

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
      # A user may not annotate the return type as secret
      rettype = to_numba_type(arg_type)
      continue
    if get_origin(arg_type) == Secret:
      assert len(get_args(arg_type)) == 1
      numba_arg = to_numba_type(get_args(arg_type)[0])
      if name == "return":
        rettype = numba_arg
        continue
      secret_args.append(idx)
      args.append(numba_arg)
    else:
      args.append(to_numba_type(arg_type))
  return args, secret_args, rettype
