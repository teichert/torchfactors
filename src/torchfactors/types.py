from typing import Any, List, Sequence, Tuple, Union

import torch
from torch import Size, Tensor

FULL_SLICE = slice(None, None, None)

NDSlice = Union[None, int, slice, Tensor, List[Any], Tuple[Any, ...]]
NDRange = Sequence[Union[range, int]]
SliceType = Union[slice, int]
ShapeType = Union[Size, Tuple[int, ...]]


class ReadOnlyView(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if (func.__name__ == '__setitem__' or
            func.__name__.endswith('_') and
                not func.__name__.endswith('__')):
            raise TypeError("you are not allowed to do in-place operations on ReadOnlyViews")
        if kwargs is None:
            kwargs = {}
        return super().__torch_function__(func, types, args, kwargs)
