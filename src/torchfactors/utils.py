import math
from itertools import chain
from typing import Tuple, cast, overload

from multimethod import multidispatch
from torch import Tensor, arange
from torch.types import Number

from .types import NDSlice, ShapeType, SliceType


@multidispatch
def _ndarange(shape: ShapeType) -> Tensor:
    r"""
    returns a sufficiently sized arange() that has been reshaped to the specified shape
    """
    return arange(math.prod(shape)).reshape(shape)


@_ndarange.register
def _(*shape: int) -> Tensor:
    return _ndarange(shape)


@overload
def ndarange(shape: ShapeType) -> Tensor: ...


@overload
def ndarange(*shape: int) -> Tensor: ...


def ndarange(*args, **kwargs):
    return _ndarange(*args, **kwargs)


def replace_negative_infinities(t: Tensor, replacement: Number = 0.0) -> Tensor:
    r"""returns a version of the given tensor without any negative infinities;
    where any -infs had bee will be inserted the given replacement"""
    return t.nan_to_num(nan=float('nan'), posinf=float('inf'), neginf=replacement)


def ndslices_overlap(lhs: NDSlice, rhs: NDSlice, shape: ShapeType) -> bool:
    r"""
    Returns true if the lhs slice overlaps at all with the rhs slice
    """
    def ints_to_ranges(ndrange):
        for a in ndrange:
            if isinstance(a, range):
                yield a
            else:
                yield range(a, a + 1)
    lhs_range = list(ints_to_ranges(as_ndrange(lhs, shape)))
    rhs_range = list(ints_to_ranges(as_ndrange(rhs, shape)))
    # doesn't overlap if any slices are empty (need to check this since one
    # slice might only be partial)
    if any(a.start == a.stop for a in chain(lhs_range, rhs_range)):
        return False
    for a, b in zip(lhs_range, rhs_range):
        if a.start >= b.stop or a.stop <= b.start:
            return False
    return True


def as_range(one_slice: slice, length: int) -> range:
    """returns a range representing the same subset of integers as the given
    slice assuming the given length"""
    return range(length)[one_slice]


def as_ndrange(ndslice: NDSlice, shape: ShapeType) -> Tuple[range, ...]:
    if isinstance(ndslice, (tuple, list)):
        if not ndslice:
            return ()
        else:
            first_slice = ndslice[0]
            if first_slice is ...:
                if ... in ndslice[1:]:
                    raise ValueError("only one set of ellises allowed in an ndslice")
                dims_left = len(shape)
                slices_left = len(ndslice) - 1
                dots_dims = dims_left - slices_left
                return (
                    tuple([range(length) for length in shape[:dots_dims]]) +
                    as_ndrange(ndslice[1:], shape[dots_dims:]))
            else:
                return (
                    (as_range(first_slice, shape[0]),) +
                    as_ndrange(ndslice[1:], shape[1:]))
    else:
        raise NotImplementedError("haven't implemented support for that kind of ndslice")


def compose_single(lhs: SliceType, rhs: SliceType, length: int):
    out = as_range(cast(slice, lhs), length)[rhs]
    return out if isinstance(out, int) else slice(out.start, out.stop, out.step)


def compose(shape: ShapeType, first: NDSlice, second: NDSlice):
    def ensure_tuple(ndslice) -> Tuple[SliceType, ...]:
        return ndslice if isinstance(ndslice, tuple) else (ndslice,)

    first = ensure_tuple(first)
    second = ensure_tuple(second)

    out = list(first) + [slice(None)] * (len(shape) - len(first))
    remaining_dims = [i for i, s in enumerate(out) if isinstance(s, slice)]
    for i, rhs in zip(remaining_dims, second):
        out[i] = compose_single(out[i], rhs, length=shape[i])
    return tuple(out)
