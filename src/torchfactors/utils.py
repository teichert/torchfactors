from __future__ import annotations

import itertools
import math
from itertools import chain
from typing import Any, List, Sized, Tuple, Union, cast, overload

import torch
from multimethod import multidispatch
from torch import Tensor, arange
from torch.utils.data.dataset import Dataset

from .types import (GeneralizedDimensionDrop, NDRangeSlice, NDSlice,
                    RangeSlice, ShapeType, SliceType)


@torch.jit.script
def sum_tensors(tensors: List[Tensor]) -> Tensor:  # pragma: no cover
    out = tensors[0]
    for t in tensors[1:]:
        out = out + t
    return out


@torch.jit.script
def min_tensors(tensors: List[Tensor]) -> Tensor:  # pragma: no cover
    out = tensors[0]
    for t in tensors[1:]:
        out = out.min(t)
    return out


def logsumexp(t: Tensor, dim: Union[None, int, List[int], Tuple[int, ...]] = None,
              keepdim=False, *, out=None):
    if dim is None:
        dim = tuple(range(len(t.shape)))
    if not isinstance(dim, int) and not dim:
        if out is not None:
            return out.copy_(t)
        else:
            return t
    else:
        return torch.logsumexp(t, dim, keepdim=keepdim, out=out)


@torch.jit.script
def expand(t: Tensor, num_batch_dims: int, index_of_t: int, target_shape: List[int]
           ):  # pragma: no cover
    # unsqueeze
    for i in range(len(target_shape) - num_batch_dims):
        if i < index_of_t:
            t = t.unsqueeze(num_batch_dims)
        elif i > index_of_t:
            t = t.unsqueeze(-1)
    out = t.expand(target_shape)
    return out


@torch.jit.script
def outer(tensors: List[Tensor], num_batch_dims: int = 0):  # pragma: no cover
    batch_shape = list(tensors[0].shape[:num_batch_dims])
    out_shape = batch_shape + [t.shape[-1] for t in tensors]
    tensors = [expand(t, num_batch_dims, i, out_shape) for i, t in enumerate(tensors)]
    out = tensors[0]
    for t in tensors[1:]:
        out = out * t
    return out


@torch.jit.script
def outer_or(tensors: List[Tensor], num_batch_dims: int):  # pragma: no cover
    batch_shape = list(tensors[0].shape[:num_batch_dims])
    out_shape = batch_shape + [t.shape[-1] for t in tensors]
    tensors = [expand(t, num_batch_dims, i, out_shape) for i, t in enumerate(tensors)]
    out = tensors[0]
    for t in tensors[1:]:
        out = out.logical_or(t)
    return out


@torch.jit.script
def outer_and(tensors: List[Tensor], num_batch_dims: int):  # pragma: no cover
    batch_shape = list(tensors[0].shape[:num_batch_dims])
    out_shape = batch_shape + [t.shape[-1] for t in tensors]
    tensors = [expand(t, num_batch_dims, i, out_shape) for i, t in enumerate(tensors)]
    out = tensors[0]
    for t in tensors[1:]:
        out = out.logical_and(t)
    return out


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
def ndarange(shape: ShapeType) -> Tensor: ...  # pragma: no cover


@overload
def ndarange(*shape: int) -> Tensor: ...  # pragma: no cover


def ndarange(*args, **kwargs):
    return _ndarange(*args, **kwargs)


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


def end(last: int, step: int):
    if step < 0:
        return last - 1
    else:
        return last + 1


def canonical_range(r: range) -> range:
    if r:
        # the stop should always be the smallest possible stop (so that it is cannonical)
        return range(r.start, end(r[-1], r.step), r.step)
    else:
        return range(0)


def canonical_range_slice(r: RangeSlice) -> RangeSlice:
    if isinstance(r, int):
        return r
    elif isinstance(r, range):
        return canonical_range(r)
    elif isinstance(r, GeneralizedDimensionDrop):
        # if they are all the same, then substitute for the single value
        # this might hide and error if the number of ints doesn't
        # actually match the required number, but this doesn't change
        # the semantics of legal ndranges
        if all(v == r.indexPerIndex[0] for v in r.indexPerIndex):
            return r.indexPerIndex[0]
        else:
            return r
    # elif isinstance(r, GeneralizedSlice):
    #     # make sure if a range could suffice, that it is used
    #     out = r.indexes
    #     if not out:
    #         return range(0)
    #     elif len(out) == 1:
    #         v = out[0]
    #         return range(v, v+1, 1)
    #     else:
    #         v1, v2 = out[:2]
    #         v_last = out[-1]
    #         step = v2 - v1
    #         # is step would be zero, we can't even make a range to compare against,
    #         if step == 0:
    #             return r
    #         cand = range(v1, end(v_last, step), step)
    #         if tuple(cand) == out:
    #             return cand
    #         else:
    #             return r
    else:
        raise TypeError("don't know how to handle that kind of slice")


def slice_to_range(one_slice: SliceType, length: int) -> RangeSlice:
    if isinstance(one_slice, slice):
        return range(length)[one_slice]
    else:
        return one_slice


def range_to_slice(one_range: Union[RangeSlice, Any]) -> SliceType:
    if isinstance(one_range, range):
        return slice(0 if one_range.start is None else one_range.start,
                     one_range.stop,
                     1 if one_range.step is None else one_range.step)
    else:
        return one_range


def canonical_ndslice(s: NDSlice, shape: ShapeType
                      ) -> NDSlice:
    if s is ...:
        return (...,)
    r = as_ndrange(s, shape)
    return tuple(
        range_to_slice(canonical_range_slice(ri))
        for ri in cast(Tuple[RangeSlice, ...], r))


def as_range(one_slice: SliceType, length: int) -> RangeSlice:
    """returns a range (or tuple of int when not representable by a range)
    representing the same subset of integers as the given
    slice assuming the given length"""
    return canonical_range_slice(slice_to_range(one_slice, length))


def as_ndrange(ndslice: NDSlice, shape: ShapeType) -> NDRangeSlice:
    r"""
    slices are used to index into a subset of a tensor, but the are not
    hashable; this converts something that could index into a hashable
    version that replaces slices with equivalent range objects (and
    lists with equivalent tuples)
    """
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
                    tuple(as_ndrange(ndslice[1:], shape[dots_dims:])))
            else:
                return (
                    (as_range(first_slice, shape[0]),) +
                    tuple(as_ndrange(ndslice[1:], shape[1:])))
    elif isinstance(ndslice, int):
        return (ndslice,)
    else:
        return (as_range(ndslice, shape[0]),)


def compose_single(lhs: SliceType, rhs: SliceType, length: int
                   ) -> SliceType:
    r"""
    returns the result of slicing into a given slice
    """
    lhs_range = as_range(lhs, length)
    out: RangeSlice
    if isinstance(lhs_range, (int, GeneralizedDimensionDrop)):
        raise ValueError("cannot index into a 0-dimensional")
    elif isinstance(rhs, GeneralizedDimensionDrop):
        out = rhs
    # elif isinstance(rhs, GeneralizedSlice):
    #     sliced = [lhs_range[r] for r in rhs.indexes
    #               if r < len(lhs_range)]
    #     out = GeneralizedSlice(sliced)
    else:
        out = lhs_range[rhs]
    # don't keep sequences if they can be replaced
    canonical = canonical_range_slice(out)
    to_slice = range_to_slice(canonical)
    return to_slice


def compose(first: NDSlice, second: NDSlice, shape: ShapeType):
    def ensure_tuple(ndslice: NDSlice) -> Tuple[SliceType, ...]:
        return ndslice if isinstance(ndslice, tuple) else (ndslice,)

    first = ensure_tuple(first)
    second = ensure_tuple(second)

    out = list(first) + [slice(None)] * (len(shape) - len(first))
    remaining_dims = [i for i, s in enumerate(out) if not isinstance(s, int)]
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
    num_batch_dims = len(binary.shape) - len(scales.shape)
    all = []
    expanded_scores_slice = (...,) + (None,) * num_config_dims
    expanded_coeffs_slice = (None,) * num_batch_dims
    for config in itertools.product(*itertools.repeat([0, 1], num_config_dims)):
        dims_to_sum = [i for i, ci in enumerate(config) if ci == 0]
        if dims_to_sum:
            coeffs = scales.sum(dims_to_sum, keepdim=True).expand_as(scales)
        else:
            coeffs = scales
        config_scores_per_batch = binary[(...,) + config]
        out = config_scores_per_batch[expanded_scores_slice] * coeffs[expanded_coeffs_slice]
        all.append(out)
    full = sum_tensors(all)
    return full


def num_trainable(m: torch.nn.Module) -> int:
    """
    (from my answer on stackoverflow: https://stackoverflow.com/a/62764464/3780389)
    returns the total number of parameters requiring grad used by `m` (only counting
    shared parameters once)
    """
    unique = {p.data_ptr(): p for p in m.parameters() if p.requires_grad}.values()
    return sum(p.numel() for p in unique)


def data_len(data: Dataset):
    """returns the length of the dataset (to please mypy)"""
    return len(cast(Sized, data))
