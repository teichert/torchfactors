from abc import ABC, ABCMeta
from dataclasses import dataclass
from enum import Enum
from typing import (Any, Callable, ClassVar, Final, Generic, Hashable,
                    Iterable, List, Optional, Sequence, Tuple, Type, TypeVar, Union,
                    get_type_hints)

import torch
from torch import Tensor
from torch._C import Value
from torchtyping import TensorDetail, TensorType  # type: ignore
from torchtyping.tensor_type import TensorTypeMixin  # type: ignore


@dataclass(frozen=True)
class Domain(TensorDetail):
    r"""Specifies the values (and corresponding integer indexes)
    that can a variable can be assigned."""

    range: Sequence

    def __iter__(self):
        return iter(self.range)

    def __repr__(self):
        return f"Domain[{self.range}]"

    @classmethod
    def tensor_repr(cls, tensor: Tensor):
        return "??"
    
    def check(self, tensor: Tensor) -> bool:
        return bool((tensor < len(self.range)).all() and (tensor >= 0).all())


@object.__new__
class Range:
    r"""
    Factory for an integer domain with a lower and upper bound.
    e.g.
    > Range(10)
    Domain(range(0, 10))
    > Range(5, 10)
    Domain(range(5, 10))
    > Range[10]
    Domain(range(0, 10))
    > Range[5:10]
    Domain(range(5, 10))

    """
    
    @staticmethod
    def __getitem__(key: Union[slice,int]) -> Domain:
        if isinstance(key, int):
            return Domain(range(key))
        return Domain(range(
            key.start if key.start is not None else 0,
            key.stop if key.stop is not None else 0,
            key.step if key.step is not None else 1))

    def __call__(*args: int) -> Domain:
        return Domain(range(*args))


class VariableType(Enum):
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
    ANNOTATED = 1 # is labeled (shouldn't be allowed during inference since should have been temporarily changed to either LATENT or CLAMPED);
                  # Note that the following would make it easy to temporarily change:
                  # observe = (mode==ANNOTATED)
                  # mode[observe] = CLAMPED
                  # mode[observe] = LATENT
                  # mode[observe] = ANNOTATED
    CLAMPED = 2   # only current value should be used
    
    # def is_free(self):
    #     return self in [VariableMode.FREE, VariableMode.FREE_WITH_MARK]

    # def is_clamped(self):
    #     return self in [VariableMode.CLAMPED, VariableMode.CLAMPED_WITH_MARK]

    # @staticmethod
    # def replace(variable: 'VariableTensor', original: 'VariableMode', substitution: 'VariableMode'):
    #     variable.tensor[variable.tensor == original] = substitution
    #     return variable



NDSlice = Union[None, int, slice, Tensor, List[Any], Tuple[Any, ...]]
FULL_SLICE = slice(None, None, None)
UsageParam = Union[torch.Tensor, VariableType]


# variables have a
# tensor a reference to the base-most tensor; a variable
# of a sub tensor will be different than a sub variable
# 

class _VariableBase:
    pass

class VariableTensor(_VariableBase, TensorTypeMixin):
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
    # TENSOR_AS_KEY: Final[ClassVar[object]] = object()
    base_cls: ClassVar[Type[_VariableBase]] = _VariableBase
    # _tensor: Tensor
    # _usage: TensorType[int8]
    # domain: Domain
    # ndslice: NDSlice
    # key: Hashable

    # def __new__(cls, *args, **kwargs):
    #     return object.__new__(cls)

    # def __init__(self, tensor: TensorType, usage: UsageParam = VariableMode.FREE_WITH_MARK, domain: Domain = Domain.OPEN,
    #              key: Hashable = TENSOR_AS_KEY):
    #     self._tensor = tensor
    #     self.domain = domain
    #     self.ndslice = FULL_SLICE
    #     if isinstance(usage, VariableMode):
    #         usage = torch.full_like(self.tensor, usage.value)
    #     self._usage = usage


    @property
    def tensor(self):
        raise ValueError("Unsupported Operation")
    
    @property
    def type(self):
        raise ValueError("Unsupported Operation")

    @property
    def domain(self):
        raise ValueError("Unsupported Operation")

    # @property
    # def tensor(self):
    #     return self._tensor[self.ndslice]
    
    # @property
    # def usage(self):
    #     return self._usage[self.ndslice]

    # def __eq__(self, other) -> bool:
    #     return self.hash_key() == other.hash_key()

    # def hash_key(self):
    #     return self.key, self.domain, self.ndslice

    # def __hash__(self) -> int:
    #     return hash(self.hash_key())

    # def sub_variable(self, sub_key: Hashable, domain: Domain=None) -> 'VariableTensor':
    #     return VariableTensor(
    #         tensor=self.tensor,
    #         domain=self.domain if domain is None else domain,
    #         key=self.key if sub_key is None else (self.key, sub_key))

    # def used_as(self, usage: UsageParam) -> 'VariableTensor':
    #     return VariableTensor(
    #         tensor=self.tensor,
    #         usage=usage,
    #         domain=self.domain,
    #         ndslice=self.ndslice,
    #         key=self.key)

    # def __getitem__(self, ndslice: NDSlice) -> 'VariableTensor':
    #     if self.ndslice is not FULL_SLICE:
    #         raise ValueError("Not allowed to double-slice a variable")
    #     return VariableTensor(
    #         tensor=self.tensor[ndslice],
    #         usage=self.usage[ndslice],
    #         domain=self.domain,
    #         ndslice=ndslice,
    #         key=self.key)

from torch import Size
from typing import cast

SliceType = Union[slice, int]
ShapeType = Union[Size, Tuple[int]]

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


class VariableBranch(VariableTensor):

    def __init__(self, root: 'Variable', ndslice: NDSlice):
        self.root = root
        self.ndslice = ndslice

    def __getitem__(self, ndslice: NDSlice) -> 'VariableTensor':
        return VariableBranch(self.root, compose(self.root.tensor.shape, self.ndslice, ndslice))

    @property
    def tensor(self):
        return self.root.tensor[self.ndslice]
    
    @property
    def type(self):
        return self.root.type[self.ndslice]

    @property
    def domain(self):
        return self.root.domain

    def __eq__(self, other) -> bool:
        return self.hash_key() == other.hash_key()

    def hash_key(self):
        return self.root, self.ndslice

    def __hash__(self) -> int:
        return hash(self.hash_key())


class Variable(VariableTensor):

    def __init__(self, tensor: Tensor, type: Union[VariableType, Tensor] = VariableType.ANNOTATED, domain: Optional[Domain] = None):
        self._tensor = tensor
        self._domain = domain
        if isinstance(type, VariableType):
            type = torch.full_like(self.tensor, type.value)
        self._type: Tensor = type

    def __getitem__(self, ndslice: NDSlice) -> VariableTensor:
        return VariableBranch(root=self, ndslice=ndslice)

    @property
    def tensor(self):
        return self._tensor
    
    @property
    def type(self):
        return self._type

    @property
    def domain(self):
        return self._domain




# class Variable(Annotated[TensorType, Domain]):

#     @staticmethod
#     def domain(obj, attr: str) -> Optional[Domain]:
#         hints = get_type_hints(obj)[attr]
#         if hasattr(hints, '__metadata__'):
#             for detail in hints.__metadata__[0]['details']:
#                 if isinstance(detail, Domain):
#                     return detail

# @dataclass
# class Utterance:
#     items: Variable[Range[4]] = None

# u = Utterance(torch.rand(100))
# u[4:5]
# from typing import get_type_hints
# print(Variable.domain(u, 'items'))


# T = TypeVar('T')

# class Model(Generic[T]):

#     _factor_generators: List[Callable[[T], Iterable[Factor]]]
#     # TODO: add in ParamsDict
#     def __init__(self):
#         self._factor_generators = []
#         self._domains = {}
#         self._parameters = ParameterDict()
#         self._modules = ModuleDict()

#     def domain(self, key: Hashable) -> Domain:
#         return self._domains.setdefault(key, Domain([]))

#     def params(self, key) -> ModelParameters:
#         return ModelParameters(self, key)

#     def factors_from(self, factor_generator: Callable[[T], Iterable[Factor]]) -> None:
#         self._factor_generators.append(factor_generator)

#     def factors(self, subject: T) -> Iterable[Factor]:
#         for gen in self._factor_generators:
#             yield from gen(subject)

#     def set_param(self, key: Hashable, value: Tensor) -> None:
#         self._parameters[f'{key}:{hash(key)}'] = Parameter(value)
    
#     def set_model(self, key: Hashable, value: Tensor) -> None:
#         self._parameters[f'{key}:{hash(key)}'] = Parameter(value)

#     def __call__(self, subject: T) -> List[Factor]:
#         return list(self.factors(subject))



# from typing import ClassVar, Optional, Union

# from torch import int8


    
# from torchtyping.tensor_type import Annotated




# from typing_extensions import _AnnotatedAlias, get_args, get_type_hints


# def subject(cls):
#     def post_init(self):
#         for k, v in get_type_hints(self, include_extras=True).items():
#             print(v)
#             if isinstance(v, VariableTensor):
#                 if v.__metadata__:
#                     meta = v.__metadata__[0]
#                     for detail in meta.get('details', []):
#                         if isinstance(detail, Domain):
#                             self.domain = detail
#                 # if hasattr(v, '__metadata__'):
#                 #     for detail in v.__metadata__[0]['details']:
#                 #         if isinstance(detail, Domain):
#                 #             return detail                
                
#     cls.__post_init__ = post_init
#     cls = dataclass(cls)
#     return cls


    
# from abc import ABC, abstractmethod

# import torch


# class Factors(ABC, Iterable['Factor']):
    
#     def __iter__(self) -> Iterator['Factor']:
#         return self.factors()

#     @abstractmethod
#     def factors(self) -> Iterator['Factor']:
#         pass


# class Factor(Factors):

#     def factors(self):
#         return [self]
    
#     # a factor needs to know how to:
#     # take in additional factors and queries;
#     # how to iterate over the product of
#     def log_einsum(self, equation):
#         pass

#     def compile_equation(self, others, queries):
#         return None
    
#     def dense(self) -> Tensor:
#         """returns a dense version"""
#         pass

# class ExactlyOneFactor(Factor):

#     def __init__(self, variables):
#         pass

# ShapeType = Union[torch.Size, Tuple[int,...], int]


# class ModelParameters:
#     """
#     associated with a single model and (should-be)
#     unique key; parameters are 
#     """

#     def __init__(self, model: 'Model', key: Hashable):
#         self.model = model
#         self.key = key
#         self.cached_tensor = None
#         self.cached_model = None

#     def namespace(self, key: Hashable) -> 'ModelParameters':
#         return ModelParameters(
#             model=self.model, key=(self.key, key))

#     def cache_tensor(self, shape: ShapeType, initialization: Optional[Callable[[Tensor], None]]= None) -> Tensor:
#         if 
#         out = torch.zeros(shape, requires_grad=True)
#         if initialization is not None:
#             initialization(out)
#         self.model.set_param(self.key, out)
#         return out
    
#     def cach_model(self, f: )
    
    

# import math


# @dataclass
# class LinearFactor(Factor):
#     variables: List[VariableTensor]
#     params: ModelParameters
#     input: Optional[Tensor] = torch.tensor([1.])
#     bias: bool = True
#     input_dimensions: int = 1

#     def log_einsum(self, equation):
#         pass

#     def compile_equation(self, others, queries):
#         return None
    
#     @cache
#     def dense(self) -> Tensor:
#         """returns a dense version"""
#         in_shapes = tuple(self.input.shape[-self.input_dimensions:])
#         out_shapes = tuple([len(t.domain) for t in self.variables])
#         m = self.params.cache_model(lambda: 
#             torch.nn.Linear(
#                 in_features=math.prod(in_shapes),
#                 out_features=math.prod(out_shapes),
#                 bias=self.bias))
#         return m(self.input)


from itertools import zip_longest

# def compose_single(first, second, length):
#     if isinstance(first, int):
#         raise TypeError("int is not composable")
#     def fix_slice(s):
#         """reutrns an equivalent slice but with Nones and negatives replaced to match length"""
#         return slice(
#             0 if s.start is None else s.start,
#             length if s.end is None else
#             length + s.end if s.end < 0 else
#             s.end,
#             1 if s.step is None else s.step)
#     first = fix_slice(first)
#     def new_stop(first, second):
#         if first is None and second is None:
#             return None
#         elif first is None:
#             return first.start + (second.stop - second.start)
#         else:
            
#     if isinstance(second, int):
#         return first.start + second * first.step
#     else:
#         return slice(
#             first.start + second.start,
#             first.start + (second.stop - second.start),
#             first.step * second.step)

    # def absolute_slice(s, length):
    #     return slice(
    #         0 if s.start is None else s.start,
    #         length if s.end is None else
    #         length + s.end if s.end < 0 else
    #         s.end,
    #         1 if s.step is None else s.step)
        
    # def compose_single_fixed(lhs, rhs):
    #     if isinstance(lhs, int):
    #         raise TypeError("'int' is not subscriptable")
    #     return slice(
    #         lhs.start + rhs.start * lhs.step,
    #         min(lhs.end, lhs.start + (rhs.stop - rhs.start),
    #         lhs.step * rhs.step)

