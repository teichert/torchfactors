import math

from multimethod import multidispatch as overload
from torch import arange

from .types import ShapeType


@overload
def ndrange(shape: ShapeType):
    r"""
    returns a sufficiently sized arange() that has been reshaped to the specified shape
    """
    return arange(math.prod(shape)).reshape(shape)


@ndrange.register
def non_tuple(*shape: int):
    return ndrange(shape)
