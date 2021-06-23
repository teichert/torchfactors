from typing import Any, List, Sequence, Tuple, Union

from torch import Size, Tensor

FULL_SLICE = slice(None, None, None)

NDSlice = Union[None, int, slice, Tensor, List[Any], Tuple[Any, ...]]
NDRange = Sequence[Union[range, int]]
SliceType = Union[slice, int]
ShapeType = Union[Size, Tuple[int, ...]]
