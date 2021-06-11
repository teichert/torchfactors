import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import (Any, Callable, Dict, Generic, Hashable, Iterable, Iterator,
                    List, Optional, Sequence, Tuple, TypeVar, Union, cast)

import torch
import typing_extensions
from multimethod import multidispatch as overload
from torch import Size, Tensor
from torch.nn import Module, ModuleDict, ParameterDict
from torch.nn.parameter import Parameter

from torchfactors import einsum

cache = lru_cache(maxsize=None)

FULL_SLICE = slice(None, None, None)

NDSlice = Union[None, int, slice, Tensor, List[Any], Tuple[Any, ...]]
SliceType = Union[slice, int]
ShapeType = Union[Size, Tuple[int, ...]]
T = TypeVar('T')


class Domain(ABC):

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __len__(self):
        pass


@dataclass(frozen=True)
class SeqDomain(Domain):
    r"""Specifies the values (and corresponding integer indexes)
    that can a variable can be assigned."""

    range: Sequence

    def __iter__(self):
        return iter(self.range)

    def __len__(self):
        return len(self.range)

    def __repr__(self):
        return f"Domain[{self.range}]"

    # for torchtyping.details if we every want to support that
    # @classmethod
    # def tensor_repr(cls, tensor: Tensor):
    #     return "??"

    # def check(self, tensor: Tensor) -> bool:
    #     return bool((tensor < len(self.range)).all() and (tensor >= 0).all())


class _OpenDomain(Domain):
    def __iter__(self):
        raise ValueError("cannot iterate over open domain")

    def __len__(self):
        raise ValueError("no size for open domain")


OPEN_DOMAIN = _OpenDomain()

# should have been able to do @object.__new__ but mypy doesn't get it


class _Range:
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
    def __getitem__(key: Union[slice, int]) -> SeqDomain:
        if isinstance(key, int):
            return SeqDomain(range(key))
        return SeqDomain(range(
            key.start if key.start is not None else 0,
            key.stop if key.stop is not None else 0,
            key.step if key.step is not None else 1))

    def __call__(*args: int) -> Domain:
        return SeqDomain(range(*args))


Range = _Range()


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
    CLAMPED = 2   # only current value should be used


# variables have a
# tensor a reference to the base-most tensor; a variable
# of a sub tensor will be different than a sub variable
#


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
    def tensor(self) -> Tensor:
        return self._get_tensor()

    @tensor.setter
    def tensor(self, value: Tensor):
        self._set_tensor(value)

    # @property
    # def usage(self) -> Tensor:
    #     return self._get_usage()

    # @usage.setter
    # def usage(self, value: Union[Tensor, VarUsage]):
    #     self._set_usage(value)

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

    def __init__(self, root: 'Var', ndslice: NDSlice):
        self.root = root
        self.ndslice = ndslice

    def __getitem__(self, ndslice: NDSlice) -> 'VarBase':
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


class Var(VarBase):
    """
    Represents a tensor wrapped with domain and usage information.

    Once VarArg Generics are available, we should be able to do this a little
    differently and easily add in more shape, etc. information about the
    tensors. For now, just annotate in comments or pass in something like
    `info=TensorType['index', int]`

    """

    @overload  # type: ignore[misc]
    def __init__(self, domain: Domain = OPEN_DOMAIN,
                 usage: Union[VarUsage, Tensor, None] = VarUsage.ANNOTATED,
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
                          usage: Union[VarUsage, Tensor, None] = VarUsage.ANNOTATED):
        self.__init__(domain, usage, tensor)  # type: ignore[misc]

    @__init__.register
    def _tensor_dom_usage(self, tensor: Tensor, domain: Domain = OPEN_DOMAIN,
                          usage: Union[VarUsage, Tensor, None] = VarUsage.ANNOTATED):
        self.__init__(domain, usage, tensor)  # type: ignore[misc]

    @__init__.register
    def _tensor_usage_dom(self, tensor: Tensor, usage: Union[VarUsage, Tensor],
                          domain: Domain = OPEN_DOMAIN):
        self.__init__(domain, usage, tensor)  # type: ignore[misc]

    @__init__.register
    def _usage_dom_tensor(self, usage: VarUsage, domain: Domain = OPEN_DOMAIN,
                          tensor: Optional[Tensor] = None):
        self.__init__(domain, usage, tensor)  # type: ignore[misc]

    @__init__.register
    def _usage_tensor_dom(self, usage: VarUsage, tensor: Tensor,
                          domain: Domain = OPEN_DOMAIN):
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


class Factors(ABC, Iterable['Factor']):
    r"""
    A collection of factors (sometimes a coherent model component is most easily
    described as a subclass of this)
    """

    def __iter__(self) -> Iterator['Factor']:
        return self.factors()

    @abstractmethod
    def factors(self) -> Iterator['Factor']:
        pass


@dataclass
class Factor:
    r"""
    A Factor has a domain which is defined by the set of variables it is
    concerned with. It is a function from any configuration of those variables
    to a mass value (i.e. we only consider non-negative values which is why we
    can operate in the log space).  To support inference among many factors, a
    factor need to know how to answer queries given other "denseable" factors
    as input and a set of (einsum style) queries to respond to.
    """
    variables: List[VarBase]

    def __iter__(self) -> Iterator[VarBase]:
        return iter(self.variables)

    def __len__(self) -> int:
        return len(self.variables)

    def query(self, others: Sequence['Factor'], *queries: Sequence[VarBase]
              ) -> Sequence[Tensor]:
        return self.queryf([f.variables for f in others], *queries)(
            others)

    # @abstractmethod
    def queryf(self, others: Sequence[Sequence[VarBase]], *queries: Sequence[VarBase]
               ) -> Callable[[Sequence['Factor']], Sequence[Tensor]]:
        raise NotImplementedError("don't know how to do queries on this")

    def dense(self) -> Tensor:
        raise NotImplementedError("don't know how to give a dense version of this")


class DensableFactor(Factor):

    @abstractmethod
    def dense(self) -> Tensor:
        pass

    def queryf(self, others: Sequence[Sequence[VarBase]], *queries: Sequence[VarBase]
               ) -> Callable[[Sequence[Factor]], Sequence[Tensor]]:
        equation = einsum.compile_generic_equation(cast(List[Sequence[VarBase]], [self.variables]) +
                                                   list(others),
                                                   queries, force_multi=True)

        def f(others: Sequence[Factor]) -> Sequence[Tensor]:
            # might be able to pull this out, but I want to make
            # sure that changes in e.g. usage are reflected
            dense = [self.dense()]
            return einsum.log_einsum(equation, dense + [f.dense() for f in others])
        return f


class ParamNamespace:
    """
    Corresponds to a particular model parameter or module which is associated
    with a unique key and the Model that actually stores everything
    """

    def __init__(self, model: 'Model', key: Hashable):
        self.model = model
        self.key = key

    def namespace(self, key: Hashable) -> 'ParamNamespace':
        return ParamNamespace(
            model=self.model, key=(self.key, key))

    def parameter(self, shape: ShapeType,
                  initialization: Optional[Callable[[Tensor], None]]
                  = torch.nn.init.kaiming_uniform_
                  ) -> Tensor:
        def gen_param():
            tensor = torch.zeros(shape)
            if initialization is not None:
                initialization(tensor)
            return Parameter(tensor)
        return self.model._get_param(self.key, check_shape=shape, default_factory=gen_param)

    def module(self, constructor: Optional[Callable[[], torch.nn.Module]] = None):
        return self.model._get_module(self.key, default_factory=constructor)


class Model(torch.nn.Module, Generic[T]):

    def __init__(self):
        super(Model, self).__init__()
        self._model_factor_generators: List[Callable[[T], Iterable[Factor]]] = []
        self._model_domains: Dict[Hashable, Domain] = {}
        self._model_parameters = ParameterDict()
        self._model_modules = ModuleDict()

    def domain(self, key: Hashable) -> Domain:
        return self._model_domains.setdefault(key, SeqDomain([]))

    def namespace(self, key) -> ParamNamespace:
        return ParamNamespace(self, key)

    def factors_from(self, factor_generator: Callable[[T], Iterable[Factor]]) -> None:
        self._model_factor_generators.append(factor_generator)

    def factors(self, subject: T) -> Iterable[Factor]:
        for gen in self._model_factor_generators:
            yield from gen(subject)

    def _get_param(self, key: Hashable, check_shape: Optional[ShapeType] = None,
                   default_factory: Optional[Callable[[], Parameter]] = None
                   ) -> Parameter:
        repr = f'{key}:{hash(key)}'
        if repr in self._model_modules:
            raise KeyError(
                "trying to get a parameter with a key "
                f"already used for a module: {repr}")
        if repr not in self._model_parameters:
            if default_factory is not None:
                param = default_factory()
                self._model_parameters[repr] = param
                return param
            else:
                raise KeyError("no param at that key and no default factory given")
        else:
            param = self._model_parameters[repr]
            if check_shape is not None and check_shape != param.shape:
                raise ValueError(
                    f"This key has already been used with different shape: "
                    f"{check_shape} vs {param.shape}")
            return param

    # def set_param(self, key: Hashable, value: Parameter, first=True) -> None:
    #     repr = f'{key}:{hash(key)}'
    #     if first and repr in self._model_parameters:
    #         raise ValueError(f"This key has already been used!: {repr}")
    #     self._model_parameters[repr] = value

    def _get_module(self, key: Hashable,
                    default_factory: Optional[Callable[[], Module]] = None
                    ) -> Module:
        repr = f'{key}:{hash(key)}'
        if repr in self._model_parameters:
            raise KeyError(
                "trying to get a module with a key "
                f"already used for a paramter: {repr}")
        if repr not in self._model_modules:
            if default_factory is not None:
                module = default_factory()
                self._model_modules[repr] = module
                return module
            else:
                raise KeyError("no module at that key and no default factory given")
        else:
            return self._model_modules[repr]

    def __call__(self, subject: T) -> List[Factor]:
        return list(self.factors(subject))


@dataclass
class Region(object):
    factor_graph_nodes: Tuple[int, ...]
    counting_number: float


@dataclass
class Strategy(object):
    regions: List[Region]
    edges: List[Tuple[int, int]]


class FactorGraph:
    def __init__(self, factors: List[Factor]):
        self.factors = factors
        num_factors = len(factors)
        variables = list(set(v for factor in factors for v in factor))
        # self.nodes = list(factors) + variables
        self.num_nodes = len(factors) + len(variables)
        self.factor_nodes = range(num_factors)
        self.variable_nodes = range(num_factors, self.num_nodes)
        self.varids = dict((v, num_factors + varid) for varid, v in enumerate(variables))
        self.neighbors: List[List[int]] = [list() for _ in range(self.num_nodes)]
        self.num_edges = 0
        for factorid, factor in enumerate(factors):
            self.num_edges += len(factor)
            for v in factor:
                varid = self.varids[v]
                self.neighbors[factorid].append(varid)
                self.neighbors[varid].append(factorid)

    def __iter__(self) -> Iterator[Factor]:
        return iter(self.factors)

    # TODO: handle queries that are not in the graph
    def query(self, *queries: Union[VarBase, Sequence[VarBase]],
              strategy=None, force_multi=False) -> Union[Sequence[Tensor], Tensor]:
        if not queries:
            queries = ((),)
        # query_list = [(q,) if isinstance(q, VarBase) else q for q in queries]
        responses: Sequence[Tensor] = []
        if len(responses) == 1 and not force_multi:
            return responses[0]
        return responses

        # dataclass_transform worked for VSCode autocomplete without single dispatch,
        # but didn't work with single dispatch nor was it recognized by mypy;
        # see:
        #
        # _T = TypeVar("_T")

        # def __dataclass_transform__(
        #     *,
        #     eq_default: bool = True,
        #     order_default: bool = False,
        #     kw_only_default: bool = False,
        #     field_descriptors: Tuple[Union[type, Callable[..., Any]], ...] = (()),
        # ) -> Callable[[_T], _T]:
        #     # If used within a stub file, the following implementation can be
        #     # replaced with "...".
        #     return lambda a: a

        # # @singledispatch
        # # # @__dataclass_transform__(order_default=True, field_descriptors=(Variable))
        # # def subject(stackable: bool = False):
        # #     def wrapped(cls: type):
        # #         cls = subject(cls)
        # #         # do other stuff
        # #         return cls
        # #     return wrapped

        # # @subject.register
        # # @__dataclass_transform__(order_default=True, field_descriptors=(Variable))
        # @__dataclass_transform__(order_default=True, field_descriptors=(Var,))
        # def subject(cls: type):
        #     setattr(cls, '__post_init__', Subject.init_variables)
        #     return dataclass(cls)


def BetheTree(graph: FactorGraph) -> Strategy:
    # TODO: add in blank factors for queries if necessary
    return Strategy(
        regions=[
            Region((factor_node, *graph.neighbors[factor_node]), 1.0)
            for factor_node in graph.factor_nodes] + [
            Region((variable_node,), 1 - len(graph.neighbors[variable_node]))
            for variable_node in graph.variable_nodes],
        edges=[(i, j) for i in range(graph.num_nodes) for j in graph.neighbors[i]])


class Subject:
    @staticmethod
    def init_variables(obj):
        cls = type(obj)
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if isinstance(attr, Var):
                property = cast(Var, getattr(obj, attr_name))
                if property.tensor is None:
                    raise ValueError(
                        "need to specify an actual tensor for every variable in the subject")
                if property.domain is OPEN_DOMAIN:
                    property._domain = attr.domain
                if property.usage is None:
                    property.usage = torch.full_like(property.tensor, attr.usage.value)

    def __post_init__(self):
        Subject.init_variables(self)

    def __init__(self, *args, **kwargs):
        raise ValueError(
            "shouldn't call this initializer: subclass Subject and use @dataclass decorator")


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


# class ExactlyOneFactor(Factor):

#     def __init__(self, variables):
#         pass


@dataclass
class LinearFactor(DensableFactor):
    variables: List[VarBase]
    params: ParamNamespace
    input: Tensor = torch.tensor([1.])
    bias: bool = True
    input_dimensions: int = 1

    def dense(self) -> Tensor:
        """returns a dense version"""
        in_shapes = tuple(self.input.shape[-self.input_dimensions:])
        out_shapes = tuple([len(t.domain) for t in self.variables])
        m = self.params.module(lambda:
                               torch.nn.Linear(
                                   in_features=math.prod(in_shapes),
                                   out_features=math.prod(out_shapes),
                                   bias=self.bias))
        return m(self.input)
