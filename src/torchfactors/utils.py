import math

from multimethod import multidispatch as overload
from torch import Tensor, arange
from torch.types import Number

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


def replace_negative_infinities(t: Tensor, replacement: Number = 0.0):
    r"""returns a version of the given tensor without any negative infinities;
    where any -infs had bee will be inserted the given replacement"""
    return t.nan_to_num(nan=float('nan'), posinf=float('inf'), neginf=replacement)
