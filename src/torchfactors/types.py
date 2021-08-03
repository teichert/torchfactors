from dataclasses import dataclass
from typing import Any, List, Tuple, Union, overload

from multimethod import multidispatch
from torch import Size


@dataclass(frozen=True)
class GeneralizedDimensionDrop:
    r"""Like a single integer index into an ndarray in that it will drop a
    dimension from the output, but allows (and requires) as separate index to be
    specified for each element of the previous surviving dimension;
    """
    indexPerIndex: Tuple[int, ...]


@multidispatch
def _gdrop(*indexes: int) -> GeneralizedDimensionDrop:
    return GeneralizedDimensionDrop(indexes)


@_gdrop.register
def _gdrop_from_list(indexes: List[int]) -> GeneralizedDimensionDrop:
    return _gdrop(*indexes)


@_gdrop.register
def _gdrop_from_tuple(indexes: Tuple[int, ...]) -> GeneralizedDimensionDrop:
    return _gdrop(*indexes)


@overload
def gdrop(*indexes: int) -> GeneralizedDimensionDrop: ...  # pragma: no cover


@overload
def gdrop(indexes: List[int]) -> GeneralizedDimensionDrop: ...  # pragma: no cover


@overload
def gdrop(indexes: Tuple[int, ...]) -> GeneralizedDimensionDrop: ...  # pragma: no cover


def gdrop(*args, **kwargs):
    return _gdrop(*args, **kwargs)


# @dataclass(frozen=True)
# class GeneralizedSlice:
#     r"""Like a slice of a dimension in that the dimeension will be retained, but
#     allows an arbitrary sequence of dimension elements to be retained (even with
#     duplicates and in a different order)"""
#     indexes: Tuple[int, ...]


# @multidispatch
# def _gslice(*indexes: int) -> GeneralizedSlice:
#     return GeneralizedSlice(indexes)


# @_gslice.register
# def _gslice_from_list(indexes: List[int]) -> GeneralizedSlice:
#     return _gslice(*indexes)


# @_gslice.register
# def _gslice_from_tuple(indexes: Tuple[int, ...]) -> GeneralizedSlice:
#     return _gslice(*indexes)


# @overload
# def gslice(*indexes: int) -> GeneralizedSlice: ...  # pragma: no cover


# @overload
# def gslice(indexes: List[int]) -> GeneralizedSlice: ...  # pragma: no cover


# @overload
# def gslice(indexes: Tuple[int, ...]) -> GeneralizedSlice: ...  # pragma: no cover


# def gslice(*args, **kwargs):
#     return _gslice(*args, **kwargs)


FULL_SLICE = slice(None, None, None)

RangeSlice = Union[range, int, GeneralizedDimensionDrop]
NDRangeSlice = Tuple[Union[RangeSlice, Any], ...]
SliceType = Union[slice, int, GeneralizedDimensionDrop]
NDSlice = Union[SliceType, Tuple[Union[SliceType, Any], ...]]
ShapeType = Union[Size, Tuple[int, ...]]
