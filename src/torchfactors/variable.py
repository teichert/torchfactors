from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Tuple, Union, cast

import torch
import typing_extensions
from multimethod import multidispatch as overload
from torch import Size, Tensor

from .domain import Domain
from .types import NDSlice, ShapeType, SliceType


class VarUsage(Enum):
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
    DEFAULT = OBSERVED


class VarBase(ABC):
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
    def tensor(self) -> Tensor:
        return self._get_tensor()

    @tensor.setter
    def tensor(self, value: Tensor):
        self._set_tensor(value)

    @property
    def domain(self) -> Domain:
        return self._get_domain()

    @abstractmethod
    def _get_tensor(self) -> Tensor:
        pass

    @abstractmethod
    def _set_tensor(self, value: Tensor):
        pass

    @abstractmethod
    def _get_usage(self) -> Tensor:
        pass

    @abstractmethod
    def _set_usage(self, value: Union[Tensor, VarUsage]):
        pass

    @abstractmethod
    def _get_domain(self) -> Domain:
        pass

    def _get_usage_(self) -> Tensor:
        return self._get_usage()

    def _set_usage_(self, value: Union[Tensor, VarUsage]):
        self._set_usage(value)

    # I wasn't allowed to make property from abstract methods
    usage = property(_get_usage_, _set_usage_)

    @abstractmethod
    def _get_original_tensor(self) -> Tensor:
        pass

    def _get_original_tensor_(self) -> Tensor:
        return self._get_original_tensor()

    original_tensor = property(_get_original_tensor_)

    def clamp_annotated(self) -> None:
        self.usage[self.usage == VarUsage.ANNOTATED] = VarUsage.CLAMPED

    def unclamp_annotated(self) -> None:
        self.usage[self.usage == VarUsage.CLAMPED] = VarUsage.ANNOTATED

    @abstractmethod
    def _get_ndslice(self) -> NDSlice:
        pass

    def _get_ndslice_(self) -> NDSlice:
        return self._get_ndslice()

    ndslice = property(_get_ndslice_)


def compose_single(lhs: SliceType, rhs: SliceType, length: int):
    out = range(length)[cast(slice, lhs)][rhs]
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


class VarBranch(VarBase):

    def __init__(self, root: Var, ndslice: NDSlice):
        self.root = root
        self.__ndslice = ndslice

    def __getitem__(self, ndslice: NDSlice) -> VarBase:
        return VarBranch(self.root, compose(self.root.tensor.shape, self.ndslice, ndslice))

    def _get_tensor(self) -> Tensor:
        return self.root.tensor[self.ndslice]

    def _set_tensor(self, value: Tensor):
        self.root.tensor[self.ndslice] = value

    def _get_usage(self) -> Tensor:
        return self.root.usage[self.ndslice]

    def _set_usage(self, value: Union[Tensor, VarUsage]):
        if isinstance(value, VarUsage):
            self.usage = torch.full_like(self.tensor, value.value)
        else:
            self.root.usage[self.ndslice] = cast(Tensor, value)

    def _get_domain(self) -> Domain:
        return self.root.domain

    def __eq__(self, other) -> bool:
        return self.hash_key() == other.hash_key()

    def hash_key(self):
        return self.root, self.ndslice

    def __hash__(self) -> int:
        return hash(self.hash_key())

    def _get_original_tensor(self) -> Tensor:
        return self.root.tensor

    def _get_ndslice(self) -> NDSlice:
        return self.__ndslice


class Var(VarBase):
    """
    Represents a tensor wrapped with domain and usage information.

    Once VarArg Generics are available, we should be able to do this a little
    differently and easily add in more shape, etc. information about the
    tensors. For now, just annotate in comments or pass in something like
    `info=TensorType['index', int]`

    """

    @overload  # type: ignore[misc]
    def __init__(self, domain: Domain = Domain.OPEN,
                 usage: Union[VarUsage, Tensor, None] = VarUsage.DEFAULT,
                 tensor: Optional[Tensor] = None,
                 info: typing_extensions._AnnotatedAlias = None):
        if tensor is not None:
            self.tensor = tensor
        self._domain = domain
        if tensor is not None:
            if usage is not None and isinstance(usage, VarUsage):
                usage = torch.full_like(self.tensor, usage.value)
            self._usage: Tensor = cast(Tensor, usage)
        self._info = info

    @__init__.register
    def _dom_tensor_usage(self, domain: Domain,
                          tensor: Tensor,
                          usage: Union[VarUsage, Tensor, None] = VarUsage.DEFAULT):
        self.__init__(domain, usage, tensor)  # type: ignore[misc]

    @__init__.register
    def _tensor_dom_usage(self, tensor: Tensor, domain: Domain = Domain.OPEN,
                          usage: Union[VarUsage, Tensor, None] = VarUsage.DEFAULT):
        self.__init__(domain, usage, tensor)  # type: ignore[misc]

    @__init__.register
    def _tensor_usage_dom(self, tensor: Tensor, usage: Union[VarUsage, Tensor],
                          domain: Domain = Domain.OPEN):
        self.__init__(domain, usage, tensor)  # type: ignore[misc]

    @__init__.register
    def _usage_dom_tensor(self, usage: VarUsage, domain: Domain = Domain.OPEN,
                          tensor: Optional[Tensor] = None):
        self.__init__(domain, usage, tensor)  # type: ignore[misc]

    @__init__.register
    def _usage_tensor_dom(self, usage: VarUsage, tensor: Tensor,
                          domain: Domain = Domain.OPEN):
        self.__init__(domain, usage, tensor)  # type: ignore[misc]

    def __getitem__(self, ndslice: NDSlice) -> VarBase:
        return VarBranch(root=self, ndslice=ndslice)

    def _get_tensor(self) -> Tensor:
        return self._tensor

    def _set_tensor(self, value: Tensor):
        self._tensor = value

    def _get_usage(self) -> Tensor:
        return self._usage

    def _set_usage(self, value: Union[Tensor, VarUsage]):
        if isinstance(value, VarUsage):
            self._usage = torch.full_like(self.tensor, value.value)
        else:
            self._usage = cast(Tensor, value)

    def _get_domain(self) -> Domain:
        return self._domain

    @staticmethod
    def pad_and_stack(batch: List['Var'], pad_value=float('nan')
                      ) -> 'Var':
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
        shapes = torch.vstack([torch.tensor(x.shape) for x in batch])
        max_shape = torch.max(shapes, 0).values
        stacked_tensors = first_tensor.new_full(
            (batch_size, *max_shape), fill_value=pad_value, dtype=dtype)
        stacked_usages = first_tensor.new_full(
            (batch_size, *max_shape), fill_value=VarUsage.PADDING.value, dtype=dtype)
        # mask = first_tensor.new_full((batch_size, *max_shape),
        # fill_value=False, dtype=torch.bool)
        for i, x in enumerate(batch):
            x_indexs = [slice(None, s) for s in x.tensor.shape]
            stacked_tensors[[i, *x_indexs]] = x.tensor
            stacked_usages[[i, *x_indexs]] = x.usage
            # mask[[i, *x_indexs]] = tensor.new_ones(tensor.shape)
        # Var(stacked, )
        return Var(first.domain, tensor=stacked_tensors, usage=stacked_usages)

    def _get_original_tensor(self) -> Tensor:
        return self.tensor

    def _get_ndslice(self) -> NDSlice:
        return (...,)
