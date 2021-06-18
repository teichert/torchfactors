from typing import Any, List, Tuple, Union

from torch import Size, Tensor

FULL_SLICE = slice(None, None, None)

NDSlice = Union[None, int, slice, Tensor, List[Any], Tuple[Any, ...]]
SliceType = Union[slice, int]
ShapeType = Union[Size, Tuple[int, ...]]
