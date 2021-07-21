from __future__ import annotations

from abc import ABC, abstractmethod
from enum import IntEnum
from functools import cached_property
from typing import (Any, Callable, List, Optional, Sequence, Union, cast,
                    overload)

import torch
import typing_extensions
from multimethod import multimethod
from torch import Size, Tensor
from torch.nn import functional as F

from .domain import Domain
from .types import GeneralizedDimensionDrop, NDSlice, ReadOnlyView, ShapeType
from .utils import as_ndrange, canonical_ndslice, compose, ndslices_cat

Tensorable = Union[Tensor, int, float, bool]


class VarUsage(IntEnum):
    r"""
    Indicates whether a variable should be used for modeling a value (FREE),
    clamped to its current value, or whether it is merely padding so that any
    touching factor should be ignored. Having "marked" versions can be
    convenient for temporarily switching a mode. For example, during training,
    there are some variables that will be clamped during one inference pass and
    not clamped during another.

    Other thoughts: posterior regularization style constraints?
    """

    PADDING = -1  # doesn't exist (any factor touching padding should be annihilated)
    LATENT = 0    # no-one knows
    # ANNOTATED means labeled (typical usage is to toggle between ANNOTATED and CLAMPED);
    ANNOTATED = 1
    # only current value should be used (but called clamped to make it easy to change back)
    CLAMPED = 2
    OBSERVED = 3  # should always be clamped
    DEFAULT = LATENT


class Var(ABC):
    """
    Represents a set of variables arranged in an n-dimensional grid. Each
    variable includes a placeholder for a value, an indicator for how the
    variable's assigned value should be interpreted (padding, unknown, or
    padding), and a domain (shared across all variables of the grid) that
    describes the possible values that the variables can be assigned and the
    corresponding numeric value (only integer, discrete supported now)

    Who uses these:
    1) A model is defined with respect to a particular "Subject" type which
       should include a VariableTensor for any quantity that may be latent or
       predicted by the model.
    2) An inference problem is given as a subject of inference along with a set
       of queries.  Analogous to einsum, each query is a (sensible) collection
       of variable slices, and the answer is the joint marginal of those
       variables. Even if variables were modeled by factors between smaller
       slices, it is still legal to include queries that deal with larger slices
       or entire VariableTensor objects so long as all slices in a group have
       the same shape [always the case if there is only one in the group]. A
       group with no variables retrieves the estimate of the partition function.
       Uniform factors will be added as necessary in order to be able to satisfy
       all queries.
    3) Ultimately, each factor is defined as a function from a variable
       configuration to a score (we use log scores for numeric precision); to
       support batching, the tensor has a tuple of variables which need to have
       the exact same shape (although they may differ in terms of domain). A
       "dense" factor, is a tensor with a dimension for each input variable. The
       size of each dimension is equal to the domain size of the respective
       variable (note that it does not depend on the dimensionality of the input
       tensor which may represent a large, multi-dimensional batch).  Structured
       (sparse) factors likewise, only implicitly have a dimension for each
       input variable. The value at each cell is the score (the multiplicand in
       the unnormalized log probability for the corresponding configuration). So
       a factor is typically created from a tuple of slices where each slice
       indicates multiple instances of the same variable across a batch. (what
       about objects that have dictionaries of variables; those won't be able to
       auto-set their domain?; I think we could have the variable look it up in
       the subject instead?)
    4) RootVariables store the tensor, domain, and mode. BranchVariables store a
       root variable and a contiguous ndslice.  They can be used to access a
       slice of the original tensor or mode. We should be able to branch from a
       branch variable just fine.  If variables are created in the model,
       those could, in principle, be accessible via the resulting factor graph.
    """

    @property
    def shape(self) -> Size:
        return self.tensor.shape

    @property
    def marginal_shape(self) -> Size:
        return Size([*self.tensor.shape, len(self.domain)])

    @property
    def read_only(self) -> bool:
        return self.origin._read_only()

    def maybe_read_only(self, t: Tensor) -> Tensor:
        if self.read_only:
            read_only = t.as_subclass(ReadOnlyView)  # type: ignore
            return read_only
        else:
            return t

    def assert_writeable(self):
        if self.read_only:
            raise TypeError("tensor and usage are read-only after caching possible or padding")

    def set_tensor(self, value: Tensorable) -> None:
        self.assert_writeable()
        self._set_tensor(value)

    def clear_cache(self):
        self.origin._clear_cache()

    @property
    def tensor(self) -> Tensor:
        t = self._get_tensor()
        return self.maybe_read_only(t)

    @tensor.setter
    def tensor(self, value: Any) -> None:
        self.set_tensor(cast(Tensorable, value))

    @property
    def domain(self) -> Domain:
        return self._get_domain()

    @abstractmethod
    def _get_tensor(self) -> Tensor: ...  # pragma: no cover

    @abstractmethod
    def __getitem__(self, ndslice: NDSlice) -> Var: ...  # pragma: no cover

    @abstractmethod
    def _set_tensor(self, value: Tensorable): ...  # pragma: no cover

    @abstractmethod
    def _get_usage(self) -> Tensor: ...  # pragma: no cover

    @abstractmethod
    def _set_usage(self, value: Union[Tensor, VarUsage]): ...  # pragma: no cover

    @abstractmethod
    def _get_domain(self) -> Domain: ...  # pragma: no cover

    @abstractmethod
    def _get_origin(self) -> TensorVar: ...  # pragma: no cover

    @property
    def origin(self) -> TensorVar:
        return self._get_origin()

    def set_usage(self, value: Union[Tensor, VarUsage]) -> None:
        self.assert_writeable()
        self._set_usage(value)

    @property
    def usage(self) -> Tensor:
        usage = self._get_usage()
        if isinstance(usage, VarUsage):
            usage = torch.full_like(self.tensor, usage, dtype=torch.int8)
            self.usage = usage
        out = self.maybe_read_only(usage)
        return out

    @usage.setter
    def usage(self, value: Any) -> None:
        self.set_usage(cast(Union[Tensor, VarUsage], value))

    @cached_property
    def _out_slice(self) -> NDSlice:
        out = ndslices_cat(self.ndslice, slice(None))
        return out

    @property
    def is_padding(self) -> Tensor:
        padding = self.origin._is_padding()
        return at(padding, self._out_slice)

    @property
    def is_possible(self) -> Tensor:
        possible = self.origin._is_possible()
        return at(possible, self._out_slice)

    def clone(self) -> Var:
        return TensorVar(
            domain=self.domain,
            usage=self.usage.clone(),
            tensor=self.tensor.clone().as_subclass(torch.Tensor),  # type: ignore
        )

    def clamp_annotated(self) -> None:
        self.usage[self.usage == VarUsage.ANNOTATED] = VarUsage.CLAMPED

    def unclamp_annotated(self) -> None:
        self.usage[self.usage == VarUsage.CLAMPED] = VarUsage.ANNOTATED

    @abstractmethod
    def _get_ndslice(self) -> NDSlice: ...  # pragma: no cover

    @property
    def ndslice(self) -> NDSlice:
        return self._get_ndslice()

    def __eq__(self, other) -> bool:
        return self.hash_key() == other.hash_key()

    def hash_key(self):
        return (id(self.origin._tensor),
                as_ndrange(self.ndslice, self.origin.tensor.shape))

    def __hash__(self) -> int:
        return hash(self.hash_key())

    @property
    def out_slice(self) -> NDSlice:
        return ndslices_cat(self.ndslice, (slice(None),))

    def flatten(self, usage: Optional[VarUsage] = None) -> Tensor:
        r"""
        Returns the flattened underlying tensor.
        If usage is specified, then only selects those indexes matching
        the specified usage.
        """
        if usage is None:
            return self.tensor.flatten()
        else:
            return self.tensor[self.usage == usage]


def at(t: Tensor, s: NDSlice, starting=0) -> Tensor:
    r"""
    Allows indexing by slices or by non-contiguous lists or tuples;
    `starting` is how many dimensions should be skipped before doing the indexing
    """
    s_tuple = s if isinstance(s, tuple) else (s,)
    first = s_tuple[0]
    rest = s_tuple[1:]
    # elipsis means to skip as many dimensions as necessary
    if first is ...:
        if not rest:
            return t
        num_left = len(rest)
        total = len(t.shape)
        out = at(t, rest, starting=total - num_left)
        return out
    # keep the previous dimensions in tact and apply the next one
    if isinstance(first, GeneralizedDimensionDrop):
        num_to_skip = starting - 1
        this_one = torch.nn.functional.one_hot(torch.tensor(first.indexPerIndex,
                                                            device=t.device),
                                               t.shape[starting]).bool()
    else:
        num_to_skip = starting
        this_one = first
    transformed = t[(slice(None),) * num_to_skip + (this_one,)]
    if not rest:
        return transformed
    elif isinstance(first, (int, GeneralizedDimensionDrop)):
        # the dimension just gets consumed
        return at(transformed, rest, starting=starting)
    else:
        return at(transformed, rest, starting=starting + 1)


class VarBranch(Var):
    r"""
    Represents a subset of a variable tensor
    """

    def __init__(self, root: TensorVar, ndslice: NDSlice):
        self.root = root
        self.__ndslice = canonical_ndslice(ndslice, root.shape)

    def _get_origin(self) -> TensorVar:
        return self.root

    def __getitem__(self, ndslice: NDSlice) -> Var:
        return VarBranch(self.root, compose(self.ndslice, ndslice,
                                            self.root.tensor.shape))

    def _get_tensor(self) -> Tensor:
        return at(self.root.tensor, self.ndslice)

    def _set_tensor(self, value: Tensorable):
        at(self.root.tensor, self.ndslice)[(...,)] = value

    def _get_usage(self) -> Tensor:
        return at(self.root.usage, self.ndslice)

    def _set_usage(self, value: Union[Tensor, VarUsage]):
        if isinstance(value, VarUsage):  # or not value.shape:
            value = torch.tensor(value, dtype=torch.int8, device=self.root.usage.device)
        at(self.root.usage, self.ndslice)[(...,)] = cast(Tensor, value.expand_as(self.tensor))

    def _get_domain(self) -> Domain:
        return self.root.domain

    def _get_ndslice(self) -> NDSlice:
        return self.__ndslice


class VarField(Var):
    r"""
    A field (in a dataclass Subject) that is a place-holder for a Variable and holds
    information that does not have to be provided on individual examples
    """

    @multimethod
    def ___init__(self,
                  domain: Domain = Domain.OPEN,
                  usage: Optional[VarUsage] = None,
                  shape: Union[Var, ShapeType, None] = None,
                  init: Callable[[ShapeType], Tensor] = torch.zeros,
                  info: typing_extensions._AnnotatedAlias = None):
        self._domain = domain
        self._usage = usage
        self._shape = shape
        self._init = init
        self._info = info

    @___init__.register
    def ___init_usage_domain(self,
                             usage: VarUsage,
                             domain: Domain = Domain.OPEN,
                             shape: Union[Var, ShapeType, None] = None,
                             init: Callable[[ShapeType], Tensor] = torch.zeros,
                             info: typing_extensions._AnnotatedAlias = None):
        self.___init__(domain=domain, usage=usage, shape=shape, init=init, info=info)

    @overload
    def __init__(self,
                 domain: Domain = Domain.OPEN,
                 usage: Optional[VarUsage] = VarUsage.DEFAULT,
                 shape: Union[Var, ShapeType, None] = None,
                 init: Callable[[ShapeType], Tensor] = torch.zeros,
                 info: typing_extensions._AnnotatedAlias = None): ...  # pragma: no cover

    @overload
    def __init__(self,
                 usage: Optional[VarUsage] = VarUsage.DEFAULT,
                 domain: Domain = Domain.OPEN,
                 shape: Union[Var, ShapeType, None] = None,
                 init: Callable[[ShapeType], Tensor] = torch.zeros,
                 info: typing_extensions._AnnotatedAlias = None): ...  # pragma: no cover

    def __init__(self, *args, **kwargs):
        self.___init__(*args, **kwargs)

    def _get_origin(self) -> TensorVar:
        raise NotImplementedError("var fields don't actually have an origin")

    def _get_tensor(self) -> Tensor:
        raise NotImplementedError("var fields don't actually have a tensor")

    def _set_tensor(self, value: Tensorable):
        raise NotImplementedError("var fields don't actually have a tensor")

    def _get_usage(self) -> Tensor:
        raise NotImplementedError("need to access _usage directly")

    def _set_usage(self, value: Union[Tensor, VarUsage]):
        raise NotImplementedError("need to access _usage directly")

    def _get_domain(self) -> Domain:
        raise NotImplementedError("need to access _domain directly")

    def _get_ndslice(self) -> NDSlice:
        raise NotImplementedError("var fields don't actually have a tensor")

    def __getitem__(self, ndslice: NDSlice) -> Var:
        raise NotImplementedError("var fields don't actually have a tensor")


def as_ndslice(shape: ShapeType) -> NDSlice:
    return tuple(
        slice(dim_size)
        for dim_size in shape)


class TensorVar(Var):
    """
    Represents a tensor wrapped with domain and usage information.

    Once VarArg Generics are available, we should be able to do this a little
    differently and easily add in more shape, etc. information about the
    tensors. For now, just annotate in comments or pass in something like
    `info=TensorType['index', int]`

    """
    @multimethod
    def ___init__(self, domain: Domain = Domain.OPEN,
                  usage: Union[VarUsage, Tensor] = VarUsage.LATENT,
                  tensor: Optional[Tensor] = None,
                  info: typing_extensions._AnnotatedAlias = None,
                  stack_shapes: Optional[Sequence[ShapeType]] = None):
        """
        when the shape is another variable object, that indicates that this variable object
        is being
        """
        self.__cached_possible: Optional[Tensor] = None
        self.__cached_padding: Optional[Tensor] = None
        self._domain = domain
        self._tensor = tensor
        # can only build usage if there is a tensor
        if self._tensor is not None:
            self.set_usage(usage)
        else:
            self._usage = usage
        self._info = info
        self._stack_shapes = stack_shapes

    @___init__.register
    def _dom_tensor_usage(self, domain: Domain,
                          tensor: Tensor,
                          usage: Union[VarUsage, Tensor] = VarUsage.DEFAULT):
        self.___init__(domain, usage, tensor)

    @___init__.register
    def _tensor_dom_usage(self, tensor: Tensor, domain: Domain = Domain.OPEN,
                          usage: Union[VarUsage, Tensor] = VarUsage.DEFAULT):
        self.___init__(domain, usage, tensor)

    @___init__.register
    def _tensor_usage_dom(self, tensor: Tensor, usage: Union[VarUsage, Tensor],
                          domain: Domain = Domain.OPEN):
        self.___init__(domain, usage, tensor)

    @___init__.register
    def _usage_dom_tensor(self, usage: VarUsage, domain: Domain = Domain.OPEN,
                          tensor: Optional[Tensor] = None):
        self.___init__(domain, usage, tensor)

    @___init__.register
    def _usage_tensor_dom(self, usage: VarUsage, tensor: Tensor,
                          domain: Domain = Domain.OPEN):
        self.___init__(domain, usage, tensor)

    @overload
    def __init__(self, domain: Domain = Domain.OPEN,
                 usage: Union[VarUsage, Tensor] = VarUsage.DEFAULT,
                 tensor: Optional[Tensor] = None,
                 info: typing_extensions._AnnotatedAlias = None,
                 stack_shapes: Optional[Sequence[ShapeType]] = None): ...  # pragma: no cover

    @overload
    def __init__(self, domain: Domain,
                 tensor: Tensor,
                 usage: Union[VarUsage, Tensor] = VarUsage.DEFAULT): ...  # pragma: no cover

    @overload
    def __init__(self, tensor: Tensor, domain: Domain = Domain.OPEN,
                 usage: Union[VarUsage, Tensor] = VarUsage.DEFAULT): ...  # pragma: no cover

    @overload
    def __init__(self, tensor: Tensor, usage: Union[VarUsage, Tensor],
                 domain: Domain = Domain.OPEN): ...  # pragma: no cover

    @overload
    def __init__(self, usage: VarUsage, domain: Domain = Domain.OPEN,
                 tensor: Optional[Tensor] = None): ...  # pragma: no cover

    @overload
    def __init__(self, usage: VarUsage, tensor: Tensor,
                 domain: Domain = Domain.OPEN): ...  # pragma: no cover

    def __init__(self, *args, **kwargs):
        self.___init__(*args, **kwargs)

    def _get_origin(self) -> TensorVar:
        return self

    def _clear_cache(self):
        self.__cached_padding = None
        self.__cached_possible = None

    def __getitem__(self, ndslice: NDSlice) -> Var:
        if self._tensor is None:
            raise ValueError("need to have a tensor before subscripting")
        return VarBranch(root=self, ndslice=ndslice)

    def _is_padding(self):
        if self.__cached_padding is None:
            self.__cached_padding = (
                self.usage == VarUsage.PADDING)[..., None].expand(self.marginal_shape)
        return self.__cached_padding

    def _is_possible(self):
        if self.__cached_possible is None:
            one_hot: Tensor = F.one_hot(self.tensor.long(), len(self.domain)).float()
            is_fixed = (
                (self.usage == VarUsage.PADDING).logical_or
                (self.usage == VarUsage.OBSERVED).logical_or
                (self.usage == VarUsage.CLAMPED))[..., None].expand_as(one_hot)
            self.__cached_possible = one_hot.where(is_fixed, torch.ones_like(one_hot)) != 0
        return self.__cached_possible

    def _get_tensor(self) -> Tensor:
        return cast(Tensor, self._tensor)

    def _set_tensor(self, value: Tensorable):
        if self._tensor is None:
            if isinstance(value, Tensor):
                self._tensor = value
            else:
                self._tensor = torch.tensor(value)  # device = ?
        else:
            at(cast(Tensor, self._tensor), self.ndslice)[(...,)] = value

    def _get_usage(self) -> Tensor:
        return cast(Tensor, self._usage)

    def _set_usage(self, value: Union[Tensor, VarUsage]) -> None:
        if isinstance(value, VarUsage) or not value.shape:
            value = torch.full_like(self.tensor, int(value), dtype=torch.int8)
        self._usage = value

    def _get_domain(self) -> Domain:
        return self._domain

    @staticmethod
    def pad_and_stack(batch: List[TensorVar], pad_value=0) -> TensorVar:
        """
        given a list of tensors with same number of dimensions but possibly
        different shapes returns: (stacked, shapes) defined as follows: stacked:
        - a single Tensor
        - `len(stacked.shape) == 1 + len(batch[0].shape)`
        - `stacked.shape[0] == len(batch)`
        - it is the result of:
            1) padding all tensors in `batch` with `pad_value` out to the
               smallest shape that can contain each element of batch
            2) stacking the resulting padded tensors
        """
        batch_size = len(batch)
        first = batch[0]
        first_tensor = first.tensor
        dtype = first_tensor.dtype
        stack_shapes = [x.shape for x in batch]
        max_shape = list(max(d) for d in zip(*stack_shapes))
        # max_shape = list(torch.max(torch.vstack(shapes), 0))
        stacked_tensors = first_tensor.new_full(
            (batch_size, *max_shape), fill_value=pad_value, dtype=dtype)
        stacked_usages = first_tensor.new_full(
            (batch_size, *max_shape), fill_value=VarUsage.PADDING.value, dtype=torch.int8)
        for i, x in enumerate(batch):
            x_indexs = [slice(None, s) for s in x.tensor.shape]
            stacked_tensors[[i, *x_indexs]] = x.tensor
            stacked_usages[[i, *x_indexs]] = x.usage
        return TensorVar(first.domain, tensor=stacked_tensors,
                         usage=stacked_usages, stack_shapes=stack_shapes)

    def unstack(self):
        if (self._tensor is None or
            self._usage is None or
            self._stack_shapes is None or
                isinstance(self._usage, VarUsage)):
            raise ValueError('Cannot unstack a non-stacked variable')
        tensors = self._tensor.unbind()
        usages = self._usage.unbind()
        return [
            TensorVar(
                domain=self.domain,
                usage=at(usage, as_ndslice(shape)),
                tensor=at(tensor, as_ndslice(shape)),
                info=self._info)
            for usage, tensor, shape
            in zip(usages, tensors, self._stack_shapes)
        ]

    def _get_ndslice(self) -> NDSlice:
        return (...,)

    def _read_only(self) -> bool:
        return self.__cached_padding is not None or self.__cached_possible is not None


def vtensor(data: Any, **kwargs):
    return TensorVar(torch.tensor(data, **kwargs))
