from __future__ import annotations

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import (Any, Callable, List, Optional, Sequence, Union, cast,
                    overload)

import torch
import typing_extensions
from multimethod import multimethod
from torch import Size, Tensor

from .domain import Domain
from .types import GeneralizedDimensionDrop, NDSlice, ShapeType
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

    Each Variable also is aware of the number of dimensions that are to be considered
    within the same element (as opposed to containing separate elements of the same batch).
    (the root stores the info and the branches cannot change it).
    """

    @property
    def shape(self) -> Size:
        return self.tensor.shape

    @property
    def marginal_shape(self) -> List[int]:
        return [*self.tensor.shape, len(self.domain)]

    # @property
    # def read_only(self) -> bool:
    #     return self.origin._read_only()

    # def maybe_read_only(self, t: Tensor) -> Tensor:
    #     if self.read_only:
    #         read_only = t.as_subclass(ReadOnlyView)  # type: ignore
    #         return read_only
    #     else:
    #         return t

    # def assert_writeable(self):
    #     if self.read_only:
    #         raise TypeError("tensor and usage are read-only after caching possible or padding")

    def set_tensor(self, value: Tensorable) -> None:
        # self.assert_writeable()
        self._set_tensor(value)

    # def clear_cache(self):
    #     self.origin._clear_cache()

    @property
    def tensor(self) -> Tensor:
        t = self._get_tensor()
        return t
        # return self.maybe_read_only(t)

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
        # self.assert_writeable()
        self._set_usage(value)

    @property
    def usage(self) -> Tensor:
        usage = self._get_usage()
        # if isinstance(usage, VarUsage):
        #     self.set_usage(usage)
        # out = self.maybe_read_only(usage)
        # return out
        return usage

    @usage.setter
    def usage(self, value: Any) -> None:
        self.set_usage(cast(Union[Tensor, VarUsage], value))

    # @cached_property
    # def _out_slice(self) -> NDSlice:
    #     out = ndslices_cat(self.ndslice, slice(None))
    #     return out

    # @property
    # def is_padding(self) -> Tensor:
    #     padding = self.origin._is_padding()
    #     return at(padding, self._out_slice)

    # @property
    # def is_possible(self) -> Tensor:
    #     possible = self.origin._is_possible()
    #     return at(possible, self._out_slice)

    @staticmethod
    def clone_tensor(t: Tensor, device: Optional[torch.device] = None) -> Tensor:
        if isinstance(device, str):
            raise TypeError("expecting actual device rather than name")
        if device is not None and t.device != device:
            return t.to(device)
        else:
            return t.clone()

    def clone(self, device=None) -> Var:
        return TensorVar(
            domain=self.domain,
            usage=self.clone_tensor(self.usage, device=device),
            tensor=self.clone_tensor(self.tensor.as_subclass(torch.Tensor),  # type: ignore
                                     device=device)  # torch bug expects wrong type
        )

    def clamp_annotated(self) -> None:
        storage = self.usage.storage()
        if storage.size() == 1:
            if storage[0] == VarUsage.ANNOTATED:  # type: ignore
                self.set_usage(VarUsage.CLAMPED)
        else:
            self.usage[self.usage == VarUsage.ANNOTATED] = VarUsage.CLAMPED

    def unclamp_annotated(self) -> None:
        storage = self.usage.storage()
        if storage.size() == 1:
            if storage[0] == VarUsage.CLAMPED:  # type: ignore
                self.set_usage(VarUsage.ANNOTATED)
        else:
            self.usage[self.usage == VarUsage.CLAMPED] = VarUsage.ANNOTATED

    @abstractmethod
    def _get_ndslice(self) -> NDSlice: ...  # pragma: no cover

    @property
    def ndslice(self) -> NDSlice:
        return self._get_ndslice()

    def __eq__(self, other) -> bool:
        return self.hash_key() == other.hash_key()

    # TODO: only look at the id of the origin (not the origin tensor which may change)
    def hash_key(self):
        return (id(self.origin._tensor),
                as_ndrange(self.ndslice, self.origin.tensor.shape))

    def __hash__(self) -> int:
        return hash(self.hash_key())

    @property
    def out_slice(self) -> NDSlice:
        return ndslices_cat(self.ndslice, (slice(None),))

    @property
    def ndims(self) -> int:
        raise NotImplementedError("no ndims for this variable")

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
        root_usage = self.root._usage
        if isinstance(value, VarUsage):  # or not value.shape:
            if isinstance(root_usage, VarUsage) and root_usage == value:
                return
            # else:
            #     root_storage = root_usage.storage()
            #     if len(root_storage) == 1 and root_storage[0] == value:
            #         return
            value = self.tensor.new_tensor(int(value))
        if isinstance(root_usage, VarUsage):  # or len(root_storage) == 1:
            # make a full version
            self.root.set_usage(self.root.usage.clone())
        expanded = cast(Tensor, value).expand_as(self.tensor)
        at(self.root.usage, self.ndslice)[(...,)] = expanded

    def _get_domain(self) -> Domain:
        return self.root.domain

    def _get_ndslice(self) -> NDSlice:
        return self.__ndslice

    @property
    def ndims(self) -> int:
        return self.origin.ndims - (len(self.origin.shape) - len(self.shape))


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
                  info: typing_extensions._AnnotatedAlias = None,
                  *, ndims: Optional[int] = None):
        self._domain = domain
        self._usage = usage
        self._shape = shape
        self._init = init
        self._info = info
        self._ndims = ndims

    @___init__.register
    def ___init_usage_domain(self,
                             usage: VarUsage,
                             domain: Domain = Domain.OPEN,
                             shape: Union[Var, ShapeType, None] = None,
                             init: Callable[[ShapeType], Tensor] = torch.zeros,
                             info: typing_extensions._AnnotatedAlias = None,
                             *, ndims: Optional[int] = None):
        self.___init__(domain=domain, usage=usage, shape=shape, init=init, info=info,
                       ndims=ndims)

    @overload
    def __init__(self,
                 domain: Domain = Domain.OPEN,
                 usage: Optional[VarUsage] = VarUsage.DEFAULT,
                 shape: Union[Var, ShapeType, None] = None,
                 init: Callable[[ShapeType], Tensor] = torch.zeros,
                 info: typing_extensions._AnnotatedAlias = None,
                 *, ndims: Optional[int] = None): ...  # pragma: no cover

    @overload
    def __init__(self,
                 usage: Optional[VarUsage] = VarUsage.DEFAULT,
                 domain: Domain = Domain.OPEN,
                 shape: Union[Var, ShapeType, None] = None,
                 init: Callable[[ShapeType], Tensor] = torch.zeros,
                 info: typing_extensions._AnnotatedAlias = None,
                 *, ndims: Optional[int] = None): ...  # pragma: no cover

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


@torch.jit.script
def expand_left(t: Tensor, shape: List[int]) -> Tensor:  # pragma: no cover
    for _ in range(len(shape) - len(t.shape)):
        t = t.unsqueeze(0)
    return t.expand(shape)


@torch.jit.script
def possibility(usage: Tensor, values: Tensor, shape: List[int],
                latent: int, annotated: int) -> Tensor:  # pragma: no cover
    unsqueezed = values.unsqueeze(-1).expand(shape)
    dom_size = shape[-1]
    all_values = expand_left(torch.arange(dom_size, device=values.device), shape)
    one_hot = unsqueezed == all_values
    free = (usage == latent).logical_or(usage == annotated)
    unsqueezed_free = free.unsqueeze(-1).expand(shape)
    # take one_hot where it is fixed, or 1 otherwise
    return one_hot.logical_or(unsqueezed_free)


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
                  stack_shapes: Optional[Sequence[ShapeType]] = None,
                  *, ndims: Optional[int] = None):
        """
        when the shape is another variable object, that indicates that this variable object
        is being
        """
        # self.__cached_possible: Optional[Tensor] = None
        # self.__cached_padding: Optional[Tensor] = None
        self._domain = domain
        self._tensor = tensor
        # can only build usage if there is a tensor
        if self._tensor is not None:
            self.set_usage(usage)
        else:
            self._usage = usage
        self._info = info
        self._stack_shapes = stack_shapes
        self._ndims = ndims

    @___init__.register
    def _dom_tensor_usage(self, domain: Domain,
                          tensor: Tensor,
                          usage: Union[VarUsage, Tensor] = VarUsage.DEFAULT,
                          *, ndims: Optional[int] = None):
        self.___init__(domain, usage, tensor, ndims=ndims)

    @___init__.register
    def _tensor_dom_usage(self, tensor: Tensor, domain: Domain = Domain.OPEN,
                          usage: Union[VarUsage, Tensor] = VarUsage.DEFAULT,
                          *, ndims: Optional[int] = None):
        self.___init__(domain, usage, tensor, ndims=ndims)

    @___init__.register
    def _tensor_usage_dom(self, tensor: Tensor, usage: Union[VarUsage, Tensor],
                          domain: Domain = Domain.OPEN,
                          *, ndims: Optional[int] = None):
        self.___init__(domain, usage, tensor, ndims=ndims)

    @___init__.register
    def _usage_dom_tensor(self, usage: VarUsage, domain: Domain = Domain.OPEN,
                          tensor: Optional[Tensor] = None,
                          *, ndims: Optional[int] = None):
        self.___init__(domain, usage, tensor, ndims=ndims)

    @___init__.register
    def _usage_tensor_dom(self, usage: VarUsage, tensor: Tensor,
                          domain: Domain = Domain.OPEN,
                          *, ndims: Optional[int] = None):
        self.___init__(domain, usage, tensor, ndims=ndims)

    @overload
    def __init__(self, domain: Domain = Domain.OPEN,
                 usage: Union[VarUsage, Tensor] = VarUsage.DEFAULT,
                 tensor: Optional[Tensor] = None,
                 info: typing_extensions._AnnotatedAlias = None,
                 stack_shapes: Optional[Sequence[ShapeType]] = None,
                 *, ndims: Optional[int] = None): ...  # pragma: no cover

    @overload
    def __init__(self, domain: Domain,
                 tensor: Tensor,
                 usage: Union[VarUsage, Tensor] = VarUsage.DEFAULT,
                 *, ndims: Optional[int] = None): ...  # pragma: no cover

    @overload
    def __init__(self, tensor: Tensor, domain: Domain = Domain.OPEN,
                 usage: Union[VarUsage, Tensor] = VarUsage.DEFAULT,
                 *, ndims: Optional[int] = None): ...  # pragma: no cover

    @overload
    def __init__(self, tensor: Tensor, usage: Union[VarUsage, Tensor],
                 domain: Domain = Domain.OPEN,
                 *, ndims: Optional[int] = None): ...  # pragma: no cover

    @overload
    def __init__(self, usage: VarUsage, domain: Domain = Domain.OPEN,
                 tensor: Optional[Tensor] = None,
                 *, ndims: Optional[int] = None): ...  # pragma: no cover

    @overload
    def __init__(self, usage: VarUsage, tensor: Tensor,
                 domain: Domain = Domain.OPEN,
                 *, ndims: Optional[int] = None): ...  # pragma: no cover

    def __init__(self, *args, **kwargs):
        self.___init__(*args, **kwargs)

    def _get_origin(self) -> TensorVar:
        return self

    # def _clear_cache(self):
    #     self.__cached_padding = None
    #     self.__cached_possible = None

    def __getitem__(self, ndslice: NDSlice) -> Var:
        if self._tensor is None:
            raise ValueError("need to have a tensor before subscripting")
        return VarBranch(root=self, ndslice=ndslice)

    @property
    def ndims(self) -> int:
        if self._ndims is None:
            return len(self.shape)
        else:
            return self._ndims

    # def _is_padding(self):
    #     # if self.__cached_padding is None:
    #     #     self.__cached_padding = (
    #     #         self.usage == VarUsage.PADDING)[..., None].expand(self.marginal_shape)
    #     # return self.__cached_padding
    #     return (self.usage == VarUsage.PADDING)[..., None].expand(self.marginal_shape)

    # def _is_possible(self):
    #     # if self.__cached_possible is None:
    #     #     one_hot: Tensor = F.one_hot(self.tensor.long(), len(self.domain)).float()
    #     #     is_fixed = (
    #     #         (self.usage == VarUsage.PADDING).logical_or
    #     #         (self.usage == VarUsage.OBSERVED).logical_or
    #     #         (self.usage == VarUsage.CLAMPED))[..., None].expand_as(one_hot)
    #     #     self.__cached_possible = one_hot.where(is_fixed, torch.ones_like(one_hot)) != 0
    #     # one_hot: Tensor = F.one_hot(self.tensor.long(), len(self.domain)).bool()
    #     # is_fixed = (
    #     #     (self.usage == VarUsage.PADDING).logical_or
    #     #     (self.usage == VarUsage.OBSERVED).logical_or
    #     #     (self.usage == VarUsage.CLAMPED))[..., None].expand_as(one_hot)
    #     # # take one_hot where it is fixed, or 1 otherwise
    #     # return one_hot.masked_fill(is_fixed.logical_not(), True)
    #     out = possibility(self.usage, self.tensor, list(self.marginal_shape),
    #                       VarUsage.LATENT, VarUsage.ANNOTATED)
    #     return out

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
        if isinstance(self._usage, VarUsage):
            if self._tensor is None:
                raise TypeError("You need to have a tensor before you can get the usage")
            else:
                return self._tensor.new_tensor(
                    int(self._usage), dtype=torch.int8).expand_as(self._tensor)
        else:
            return self._usage

    def _set_usage(self, value: Union[Tensor, VarUsage]) -> None:
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
                         usage=stacked_usages, stack_shapes=stack_shapes,
                         ndims=first.ndims)

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
                info=self._info, ndims=self.ndims)
            for usage, tensor, shape
            in zip(usages, tensors, self._stack_shapes)
        ]

    def _get_ndslice(self) -> NDSlice:
        return (...,)

    # def _read_only(self) -> bool:
    #     return self.__cached_padding is not None or self.__cached_possible is not None


def vtensor(data: Any, **kwargs):
    return TensorVar(torch.tensor(data, **kwargs))
