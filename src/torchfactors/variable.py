from __future__ import annotations

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import (Any, Callable, List, Optional, Sequence, Union, cast,
                    overload)

import torch
import typing_extensions
from multimethod import multimethod
from torch import Size, Tensor
from torch.nn import functional as F

from .domain import Domain
from .types import NDSlice, ShapeType
from .utils import as_ndrange, compose, ndslices_cat, ndslices_overlap

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

    PADDING = -1  # doesn't exist
    LATENT = 0    # no-one knows
    # is labeled (shouldn't be allowed during inference since should have been
    # temporarily changed to either LATENT or CLAMPED);
    ANNOTATED = 1
    # Note that the following would make it easy to temporarily change:
    # observe = (mode==ANNOTATED)
    # mode[observe] = CLAMPED
    # mode[observe] = LATENT
    # mode[observe] = ANNOTATED
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

    def set_tensor(self, value: Tensorable) -> None:
        self._set_tensor(value)

    @property
    def tensor(self) -> Tensor:
        return self._get_tensor()

    @tensor.setter
    def tensor(self, value: Any) -> None:
        self.set_tensor(cast(Tensorable, value))

    # # @property
    # def get_tensor(self) -> Tensor:
    #     return self._get_tensor()

    # # @tensor.setter
    # def set_tensor(self, value: Tensorable):
    #     self._set_tensor(value)

    # def set_tensor(self, value: Tensorable):
    #     self._set_tensor(value)

    # tensor = property(get_tensor, set_tensor)

    @property
    def domain(self) -> Domain:
        return self._get_domain()

    @abstractmethod
    def _get_tensor(self) -> Tensor: ...

    @abstractmethod
    def __getitem__(self, ndslice: NDSlice) -> Var: ...

    @abstractmethod
    def _set_tensor(self, value: Tensorable): ...

    @abstractmethod
    def _get_usage(self) -> Tensor: ...

    @abstractmethod
    def _set_usage(self, value: Union[Tensor, VarUsage]): ...

    @abstractmethod
    def _get_domain(self) -> Domain: ...

    @abstractmethod
    def _get_origin(self) -> TensorVar: ...

    def overlaps(self, other: Var) -> bool:
        return (
            self.origin is other.origin and
            ndslices_overlap(self.ndslice, other.ndslice, self.origin.shape))

    @property
    def origin(self) -> TensorVar:
        return self._get_origin()

    def set_usage(self, value: Union[Tensor, VarUsage]) -> None:
        self._set_usage(value)

    @property
    def usage(self) -> Tensor:
        out = self._get_usage()
        if isinstance(out, VarUsage):
            out = torch.full_like(self.tensor, out)
            self.usage = out
            # raise TypeError(
            #     "your variable needs tensor before you can access the tensor-based usage!")
        return out

    @usage.setter
    def usage(self, value: Any) -> None:
        self.set_usage(cast(Union[Tensor, VarUsage], value))

    @abstractmethod
    def _get_original_tensor(self) -> Tensor: ...

    @property
    def original_tensor(self) -> Tensor:
        return self._get_original_tensor()

    @property
    def is_padding(self) -> Tensor:
        return (self.usage == VarUsage.PADDING)[..., None].expand(self.marginal_shape)

    @property
    def is_possible(self) -> Tensor:
        one_hot: Tensor = F.one_hot(self.tensor.long(), len(self.domain)).float()
        is_fixed = (
            (self.usage == VarUsage.PADDING).logical_or
            (self.usage == VarUsage.OBSERVED).logical_or
            (self.usage == VarUsage.CLAMPED))[..., None].expand_as(one_hot)
        return one_hot.where(is_fixed, torch.ones_like(one_hot)) != 0

    # @property
    # def usage_mask(self) -> Tensor:
    #     r"""
    #     Returns a tensor of the same shape as the variable marginal would be
    #     with (log) 1 for allowed, 0 for not-allowed, and nan for padding.

    #     The idea is that any factors touch any padding variable should be as if
    #     they were gone, so the nans are used here and then nans are replaced
    #     with 1's after combining all variables
    #     """
    #     # ones, one_hots, where one and padding is nan
    #     one_hot: Tensor = F.one_hot(self.tensor.long(), len(self.domain)).float()
    #     expanded_usage = self.usage[..., None].expand_as(one_hot)
    #     return torch.where(
    #         (expanded_usage == VarUsage.OBSERVED).
    #         logical_or(expanded_usage == VarUsage.CLAMPED).
    #         logical_or(expanded_usage == VarUsage.PADDING),
    #         torch.where(one_hot.logical_and(expanded_usage == VarUsage.PADDING),
    #                     torch.full_like(one_hot, float('nan')),
    #                     one_hot),
    #         torch.ones_like(one_hot)).log()
    #     # return self._get_original_tensor()

    def clamp_annotated(self) -> None:
        self.usage[self.usage == VarUsage.ANNOTATED] = VarUsage.CLAMPED

    def unclamp_annotated(self) -> None:
        self.usage[self.usage == VarUsage.CLAMPED] = VarUsage.ANNOTATED

    @abstractmethod
    def _get_ndslice(self) -> NDSlice: ...

    @property
    def ndslice(self) -> NDSlice:
        return self._get_ndslice()

    def __eq__(self, other) -> bool:
        return self.hash_key() == other.hash_key()

    # TODO: make an ndrange which can be hashed and use that instead
    def hash_key(self):
        return (id(self.original_tensor),
                as_ndrange(self.ndslice, self.original_tensor.shape))

    def __hash__(self) -> int:
        return hash(self.hash_key())

    @property
    def out_slice(self) -> NDSlice:
        return ndslices_cat(self.ndslice, (slice(None),))


class VarBranch(Var):
    r"""
    Represents a subset of a variable tensor
    """

    def __init__(self, root: TensorVar, ndslice: NDSlice):
        self.root = root
        self.__ndslice = ndslice

    def _get_origin(self) -> TensorVar:
        return self.root

    def __getitem__(self, ndslice: NDSlice) -> Var:
        return VarBranch(self.root, compose(self.root.tensor.shape, self.ndslice, ndslice))

    def _get_tensor(self) -> Tensor:
        return self.root.tensor[self.ndslice]

    def _set_tensor(self, value: Tensorable):
        self.root.tensor[self.ndslice] = value

    def _get_usage(self) -> Tensor:
        return self.root.usage[self.ndslice]

    def _set_usage(self, value: Union[Tensor, VarUsage]):
        if isinstance(value, VarUsage):  # or not value.shape:
            value = torch.tensor(value)
        self.root.usage[self.ndslice] = cast(Tensor, value.expand_as(self.tensor))

    def _get_domain(self) -> Domain:
        return self.root.domain

    def _get_original_tensor(self) -> Tensor:
        return self.root.tensor

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

    # @___init__.register
    # def ___init_domain_shape(self,
    #                          domain: Domain,
    #                          shape: Union[Var, ShapeType],
    #                          usage: Optional[VarUsage] = VarUsage.DEFAULT,
    #                          init: Callable[[ShapeType], Tensor] = torch.zeros,
    #                          info: typing_extensions._AnnotatedAlias = None):
    #     self.___init__(domain=domain, usage=usage, shape=shape, init=init, info=info)

    @___init__.register
    def ___init_usage_domain(self,
                             usage: VarUsage,
                             domain: Domain = Domain.OPEN,
                             shape: Union[Var, ShapeType, None] = None,
                             init: Callable[[ShapeType], Tensor] = torch.zeros,
                             info: typing_extensions._AnnotatedAlias = None):
        self.___init__(domain=domain, usage=usage, shape=shape, init=init, info=info)

    # @___init__.register
    # def ___init_usage_shape(self,
    #                         usage: VarUsage,
    #                         shape: Union[Var, ShapeType],
    #                         domain: Domain = Domain.OPEN,
    #                         init: Callable[[ShapeType], Tensor] = torch.zeros,
    #                         info: typing_extensions._AnnotatedAlias = None):
    #     self.___init__(domain=domain, usage=usage, shape=shape, init=init, info=info)

    # @___init__.register
    # def ___init_shape_usage(self,
    #                         shape: Union[Var, ShapeType],
    #                         usage: Optional[VarUsage] = VarUsage.DEFAULT,
    #                         domain: Domain = Domain.OPEN,
    #                         init: Callable[[ShapeType], Tensor] = torch.zeros,
    #                         info: typing_extensions._AnnotatedAlias = None):
    #     self.___init__(domain=domain, usage=usage, shape=shape, init=init, info=info)

    # @___init__.register
    # def ___shape_domain(self,
    #                     shape: Union[Var, ShapeType],
    #                     domain: Domain,
    #                     usage: Optional[VarUsage] = VarUsage.DEFAULT,
    #                     init: Callable[[ShapeType], Tensor] = torch.zeros,
    #                     info: typing_extensions._AnnotatedAlias = None):
    #     self.___init__(domain=domain, usage=usage, shape=shape, init=init, info=info)

    @overload
    def __init__(self,
                 domain: Domain = Domain.OPEN,
                 usage: Optional[VarUsage] = VarUsage.DEFAULT,
                 shape: Union[Var, ShapeType, None] = None,
                 init: Callable[[ShapeType], Tensor] = torch.zeros,
                 info: typing_extensions._AnnotatedAlias = None): ...

    # @overload
    # def __init__(self,
    #              domain: Domain = Domain.OPEN,
    #              shape: Union[Var, ShapeType, None] = None,
    #              usage: Optional[VarUsage] = VarUsage.DEFAULT,
    #              init: Callable[[ShapeType], Tensor] = torch.zeros,
    #              info: typing_extensions._AnnotatedAlias = None): ...

    @overload
    def __init__(self,
                 usage: Optional[VarUsage] = VarUsage.DEFAULT,
                 domain: Domain = Domain.OPEN,
                 shape: Union[Var, ShapeType, None] = None,
                 init: Callable[[ShapeType], Tensor] = torch.zeros,
                 info: typing_extensions._AnnotatedAlias = None): ...

    # @overload
    # def __init__(self,
    #              usage: Optional[VarUsage] = VarUsage.DEFAULT,
    #              shape: Union[Var, ShapeType, None] = None,
    #              domain: Domain = Domain.OPEN,
    #              init: Callable[[ShapeType], Tensor] = torch.zeros,
    #              info: typing_extensions._AnnotatedAlias = None): ...

    # @overload
    # def __init__(self,
    #              shape: Union[Var, ShapeType, None] = None,
    #              usage: Optional[VarUsage] = VarUsage.DEFAULT,
    #              domain: Domain = Domain.OPEN,
    #              init: Callable[[ShapeType], Tensor] = torch.zeros,
    #              info: typing_extensions._AnnotatedAlias = None): ...

    # @overload
    # def __init__(self,
    #              shape: Union[Var, ShapeType, None] = None,
    #              domain: Domain = Domain.OPEN,
    #              usage: Optional[VarUsage] = VarUsage.DEFAULT,
    #              init: Callable[[ShapeType], Tensor] = torch.zeros,
    #              info: typing_extensions._AnnotatedAlias = None): ...

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

    def _get_original_tensor(self) -> Tensor:
        raise NotImplementedError("var fields don't actually have a tensor")

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
                 stack_shapes: Optional[Sequence[ShapeType]] = None): ...

    @overload
    def __init__(self, domain: Domain,
                 tensor: Tensor,
                 usage: Union[VarUsage, Tensor] = VarUsage.DEFAULT): ...

    @overload
    def __init__(self, tensor: Tensor, domain: Domain = Domain.OPEN,
                 usage: Union[VarUsage, Tensor] = VarUsage.DEFAULT): ...

    @overload
    def __init__(self, tensor: Tensor, usage: Union[VarUsage, Tensor],
                 domain: Domain = Domain.OPEN): ...

    @overload
    def __init__(self, usage: VarUsage, domain: Domain = Domain.OPEN,
                 tensor: Optional[Tensor] = None): ...

    @overload
    def __init__(self, usage: VarUsage, tensor: Tensor,
                 domain: Domain = Domain.OPEN): ...

    def __init__(self, *args, **kwargs):
        self.___init__(*args, **kwargs)

    def _get_origin(self) -> TensorVar:
        return self

    def __getitem__(self, ndslice: NDSlice) -> Var:
        if self._tensor is None:
            raise ValueError("need to have a tensor before subscripting")
        return VarBranch(root=self, ndslice=ndslice)

    def _get_tensor(self) -> Tensor:
        return cast(Tensor, self._tensor)

    def _set_tensor(self, value: Tensorable):
        if self._tensor is None:
            if isinstance(value, Tensor):
                self._tensor = value
            else:
                self._tensor = torch.tensor(value)
        else:
            cast(Tensor, self._tensor)[self.ndslice] = value

    def _get_usage(self) -> Tensor:
        return cast(Tensor, self._usage)

    def _set_usage(self, value: Union[Tensor, VarUsage]) -> None:
        if isinstance(value, VarUsage) or not value.shape:
            value = torch.full_like(self.tensor, int(value))
        self._usage = value

    def _get_domain(self) -> Domain:
        return self._domain

    @staticmethod
    def pad_and_stack(batch: List['TensorVar'], pad_value=0,
                      ) -> 'TensorVar':
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
        shapes = [torch.tensor(shape) for shape in stack_shapes]
        max_shape = torch.max(torch.vstack(shapes), 0).values
        stacked_tensors = first_tensor.new_full(
            (batch_size, *max_shape), fill_value=pad_value, dtype=dtype)
        stacked_usages = first_tensor.new_full(
            (batch_size, *max_shape), fill_value=VarUsage.PADDING.value, dtype=torch.int)
        # mask = first_tensor.new_full((batch_size, *max_shape),
        # fill_value=False, dtype=torch.bool)
        for i, x in enumerate(batch):
            x_indexs = [slice(None, s) for s in x.tensor.shape]
            stacked_tensors[[i, *x_indexs]] = x.tensor
            stacked_usages[[i, *x_indexs]] = x.usage
            # mask[[i, *x_indexs]] = tensor.new_ones(tensor.shape)
        # Var(stacked, )
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
                usage=usage[as_ndslice(shape)],
                tensor=tensor[as_ndslice(shape)],
                info=self._info)
            for usage, tensor, shape
            in zip(usages, tensors, self._stack_shapes)
        ]

    def _get_original_tensor(self) -> Tensor:
        return self.tensor

    def _get_ndslice(self) -> NDSlice:
        return (...,)


def vtensor(data: Any, **kwargs):
    return TensorVar(torch.tensor(data, **kwargs))
