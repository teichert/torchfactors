import itertools
import math
from itertools import chain
from typing import Tuple, cast, overload

import torch
from multimethod import multidispatch
from opt_einsum import contract  # type: ignore
from torch import Tensor, arange

from .types import NDSlice, ShapeType, SliceType


@multidispatch
def _ndarange(shape: ShapeType) -> Tensor:
    r"""
    returns a sufficiently sized arange() that has been reshaped to the specified shape
    """
    return arange(math.prod(shape)).reshape(shape)


def outer(*tensors: Tensor, num_batch_dims=0):
    return contract(*[arg
                      for t in tensors
                      for arg in [t, [*range(num_batch_dims), id(t)]]],
                    [*range(num_batch_dims), *list(map(id, tensors))],
                    backend='torch')

    # return torch.stack(torch.meshgrid(*tensors), 0).prod(0)


@_ndarange.register
def _(*shape: int) -> Tensor:
    return _ndarange(shape)


@overload
def ndarange(shape: ShapeType) -> Tensor: ...


@overload
def ndarange(*shape: int) -> Tensor: ...


def ndarange(*args, **kwargs):
    return _ndarange(*args, **kwargs)


# def replace_negative_infinities(t: Tensor, replacement: Number = 0.0) -> Tensor:
#     r"""returns a version of the given tensor without any negative infinities;
#     where any -infs had bee will be inserted the given replacement"""
#     return t.nan_to_num(nan=float('nan'), posinf=float('inf'), neginf=replacement)


def ndslices_cat(lhs: NDSlice, rhs: NDSlice) -> NDSlice:
    r"""returns the concatenation of two ndslices"""
    if not isinstance(lhs, (list, tuple)):
        lhs = (lhs,)
    if not isinstance(rhs, (list, tuple)):
        rhs = (rhs,)
    return tuple(lhs) + tuple(rhs)


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

# TODO: instead, just get the overlap
# def ndslices_overlap(lhs: NDSlice, rhs: NDSlice, shape: ShapeType) -> bool:
#     r"""
#     Returns true if the lhs slice overlaps at all with the rhs slice
#     """
#     def ints_to_ranges(ndrange):
#         for a in ndrange:
#             if isinstance(a, range):
#                 yield a
#             else:
#                 yield range(a, a + 1)
#     lhs_range = list(ints_to_ranges(as_ndrange(lhs, shape)))
#     rhs_range = list(ints_to_ranges(as_ndrange(rhs, shape)))
#     # doesn't overlap if any slices are empty (need to check this since one
#     # slice might only be partial)
#     if any(a.start == a.stop for a in chain(lhs_range, rhs_range)):
#         return False
#     for a, b in zip(lhs_range, rhs_range):
#         if a.start >= b.stop or a.stop <= b.start:
#             return False
#     return True


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


def stereotype(scales: Tensor, binary: Tensor) -> Tensor:
    r"""
    given a scales tensor with D dimensions, and another tensor with B + D
    dimensions and the last D dimensions each of size exactly 2, returns a
    B+D dimensional tensor where the last D dimensions match the dimension
    sizes of `scales`.  The first B dimensions correnspond to separate
    elements of the batch.  Within the same element of the batch,
    the output cell is a linear interpolation of the cells of
    the corresponding slice of `binary`.

    In other words, the output is an interpolation over the cells
    in the binary tensor, each value being multiplied by the
    scales tensor after summing out dimensions of the latter corresponding
    to zeros in the binary configuration and then expanding it appropriately.
    """
    num_config_dims = len(scales.shape)
    all = []
    for config in itertools.product(*itertools.repeat([0, 1], num_config_dims)):
        dims_to_sum = [i for i, ci in enumerate(config) if ci == 0]
        coeffs = scales.sum(dims_to_sum, keepdim=True).expand_as(scales)
        out = binary[..., config] * coeffs
        all.append(out)
    full = torch.stack(all, -1).sum(-1)
    return full
