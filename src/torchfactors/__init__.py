import copy
import math
from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
from functools import cached_property, lru_cache
from itertools import chain
from typing import (Any, Callable, ClassVar, Dict, FrozenSet, Generic,
                    Hashable, Iterable, Iterator, List, Optional, Sequence,
                    Tuple, TypeVar, Union, cast)

import torch
import typing_extensions
from multimethod import multidispatch as overload
from torch import Size, Tensor
from torch.nn import Module, ModuleDict, ParameterDict
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset

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
    # only current value should be used (but called clamped to make it easy to change back)
    CLAMPED = 2
    OBSERVED = 3  # should always be clamped
    DEFAULT = OBSERVED

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

    def clamp_annotated(self) -> None:
        self.usage[self.usage == VarUsage.ANNOTATED] = VarUsage.CLAMPED

    def unclamp_annotated(self) -> None:
        self.usage[self.usage == VarUsage.CLAMPED] = VarUsage.ANNOTATED


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
    def _tensor_dom_usage(self, tensor: Tensor, domain: Domain = OPEN_DOMAIN,
                          usage: Union[VarUsage, Tensor, None] = VarUsage.DEFAULT):
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

    @staticmethod
    def pad_and_stack(batch: List['Var'], pad_value=float('nan')
                      ) -> 'Var':
        """
        given a list of tensors with same number of dimensions but possibly different shapes returns:
        (stacked, shapes) defined as follows:
        stacked:
        - a single Tensor
        - `len(stacked.shape) == 1 + len(batch[0].shape)`
        - `stacked.shape[0] == len(batch)`
        - it is the result of:
            1) padding all tensors in `batch` with `pad_value` out to the smallest shape that can contain each element of batch
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
            (batch_size, *max_shape), fill_value=VarUsage.PADDING, dtype=dtype)
        # mask = first_tensor.new_full((batch_size, *max_shape), fill_value=False, dtype=torch.bool)
        for i, x in enumerate(batch):
            x_indexs = [slice(None, s) for s in x.tensor.shape]
            stacked_tensors[[i, *x_indexs]] = x.tensor
            stacked_usages[[i, *x_indexs]] = x.usage
            # mask[[i, *x_indexs]] = tensor.new_ones(tensor.shape)
        # Var(stacked, )
        return Var(first.domain, tensor=stacked_tensors, usage=stacked_usages)


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

    def free_energy(self, other_energy: Sequence['Factor'], messages: Sequence['Factor']
                    ) -> Tensor:
        """
        an estimate of the contribution of this factor to the -log z;
        it is the entropy minus the average energy
        under a distribution given by the normalized product of all energy
        factors and all messages
        """
        raise NotImplementedError("don't know how to do queries on this")

    def dense(self) -> Tensor:
        raise NotImplementedError("don't know how to give a dense version of this")

    @cached_property
    def out_shape(self):
        return tuple([len(t.domain) for t in self.variables])

    @cached_property
    def shape(self):
        return tuple(*self.batches_shape, *self.out_shape)

    @cached_property
    def cells(self):
        return math.prod(self.shape)

    @cached_property
    def batches_shape(self):
        # should be the same for all variables (maybe worth checking?)
        first = self.variables[0]
        return tuple(first.tensor.shape[:-1])


class DensableFactor(Factor):

    @abstractmethod
    def dense_(self) -> Tensor:
        pass

    @property
    def dense(self) -> Tensor:
        d = self.dense_()
        # I only care about fixing the output here (don't care about observed
        # inputs since those have already been clamped and set to nan)
        # excluded_mask is anything that is clamped or observed and not the
        # current value as well as anything that is padded and not 0
        # TODO: finish this
        d[excluded_mask] = float('-inf')
        # clamped_mask is anything that is clamped or observed and is the target
        d[clamped_mask] = 0.0
        # padded_mask is anything that is padded and is 0
        d[padded_mask] = float('nan')
        return d

    def queryf(self, others: Sequence[Sequence[VarBase]], *queries: Sequence[VarBase]
               ) -> Callable[[Sequence[Factor]], Sequence[Tensor]]:
        equation = einsum.compile_generic_equation(cast(List[Sequence[VarBase]], [self.variables]) +
                                                   list(others),
                                                   queries, force_multi=True)

        def f(others: Sequence[Factor]) -> Sequence[Tensor]:
            # might be able to pull this out, but I want to make
            # sure that changes in e.g. usage are reflected
            dense = [self.dense()]
            # any nans in any factor should be treated as a log(1)
            # meaning that it doesn't impact the product
            return einsum.log_einsum(equation, dense + [f.dense() for f in others])
        return f

    @staticmethod
    def normalize(self, variables: Sequence[VarBase], tensor: Tensor) -> Tensor:
        num_dims = len(tensor.shape)
        num_batch_dims = len(tensor.shape) - len(variables)

        # normalize by subtracting out the sum of the last |V| dimensions
        variable_dims = list(range(num_batch_dims, num_dims))
        normalizer = torch.logsumexp(tensor, dim=variable_dims)
        tensor -= normalizer[[...] + [None] * (num_dims - num_batch_dims)]
        return tensor

    def free_energy(self, other_energy: Sequence['Factor'], messages: Sequence['Factor']
                    ) -> Tensor:
        # TODO?: there is a way to do this with expectation semiring that would be general to
        # non-denseables
        variables = list(set(v
                             for f in [self, *other_energy]
                             for v in f.variables))
        log_belief = self.query([*other_energy, *messages], variables)[0]
        log_belief = DensableFactor.normalize(variables, log_belief)
        # positives = torch.logsumexp(log_belief.clamp_min(0) +
        #                             torch.where(log_belief >= 0,log_belief.clamp_min(0).log(), 0.),
        #                             dim=variable_dims)
        # negatives = torch.logsumexp(-log_belief.clamp_max(0) +
        #                             torch.where(log_belief < 0, (-log_belief.clamp_max(0)).log(), 0.),
        #                             dim=variable_dims)
        # entropy = torch.logsumexp(log_belief * log_belief.log(), dim=variable_dims)
        log_potentials, = self.query(other_energy, variables)
        belief = log_belief.exp()
        entropy = torch.sum(belief * log_belief, dim=variable_dims)
        avg_energy = torch.sum(belief * log_potentials)
        return entropy - avg_energy


@dataclass
class TensorFactor(DensableFactor):
    _tensor: InitVar[Optional[Tensor]] = None

    def dense(self):
        return self.tensor

    def __post_init__(self, tensor: Optional[Tensor]):
        if tensor is not None:
            self.__tensor = tensor
        else:
            self.__tensor = torch.zeros(
                *[len(v.domain) for v in self.variables])

    @property
    def tensor(self):
        # in here, we need to only allow the values consistent
        # with clamped or observed variables (competitors go to log(0));
        # and then we need to take any padded inputs
        # as log(1)
        #
        return self.__tensor


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


class FactorGraph:
    """
    """

    def __init__(self, factors: List[Factor]):
        # factors come first, then variables
        self.factors = factors
        self.num_factors = len(factors)
        self.variables = list(set(v for factor in factors for v in factor))
        # self.nodes = list(factors) + variables
        self.num_nodes = len(factors) + len(self.variables)
        self.factor_nodes = range(self.num_factors)
        self.variable_nodes = range(self.num_factors, self.num_nodes)
        self.varids = dict((v, self.num_factors + varid) for varid, v in enumerate(self.variables))
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

    def full_region(self, node_ids: Sequence[int]
                    ) -> List[int]:
        return list(set([
            node_id
            for node_id in node_ids
            if node_id >= self.num_factors
        ] + [
            v
            for node_id in node_ids if node_id < self.num_factors
            for v in self.neighbors[node_id]
        ]))

    def region_factors(self, node_ids: Sequence[int]
                       ) -> List[Factor]:
        return [
            self.factors[node_id]
            for node_id in node_ids
            if node_id < self.num_factors
        ]

    def region_variables(self, node_ids: Sequence[int]
                         ) -> List[VarBase]:
        return [
            self.variables[node_id - self.num_factors]
            for node_id in self.full_region(node_ids)
        ]

    # TODO: handle queries that are not in the graph
    # should I be normalizing? probably
    def query(self, *queries: Optional[VarBase],
              strategy=None, force_multi=False) -> Union[Sequence[Tensor], Tensor]:
        if strategy is None:
            strategy = BetheTree(self)
        if not queries:
            queries = (None,)
        # query_list = [(q,) if isinstance(q, VarBase) else q for q in queries]
        bp = BPInference(self, strategy)
        bp.run()
        responses: Sequence[Tensor] = tuple(
            bp.belief(query) if query is not None else bp.logz() for query in queries)
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


@dataclass
class Region(object):
    factor_graph: FactorGraph
    factor_graph_nodes: Tuple[int, ...]
    counting_number: float

    @cached_property
    def variables(self) -> Sequence[VarBase]:
        return self.factor_graph.region_variables(self.factor_graph_nodes)

    @cached_property
    def factors(self) -> Sequence[Factor]:
        return self.factor_graph.region_factors(self.factor_graph_nodes)

    @cached_property
    def factor_set(self) -> FrozenSet[Factor]:
        return frozenset(self.factors)

    def query(self, others: Sequence[Factor],
              *queries: Sequence[VarBase], exclude: Optional['Region'] = None):
        return self.queryf(others, *queries, exclude=exclude)

    @cache
    def queryf(self, others: Sequence[Factor], exclude: Optional['Region'],
               *queries: Sequence[VarBase]
               ) -> Callable[[], Sequence[Tensor]]:
        # factors appearing in the excluded region are excluded note that the
        # first factor (in the region) touching the most variables determines
        # how the inference is done (if this needs to be modified, then have
        # your strategy create subclasses of region that override this behavior)
        surviving_factors = list((self.factor_set - exclude.factor_set)
                                 if exclude is not None else self.factor_set)
        _, ix, controller = max((len(f.variables), i, f) for i, f in enumerate(surviving_factors))
        input_factors = list(surviving_factors[:ix]) + list(surviving_factors[ix:]) + list(others)
        wrapped = controller.queryf([f.variables for f in input_factors], *queries)

        def f():
            return wrapped(input_factors)
        return f

    # TODO: here
    # def free_energy(self, messages: Sequence['Factor']) -> Tensor:
    #     _, ix, controller = max((len(f.variables), i, f) for i, f in enumerate(self.factors))


@dataclass
class Strategy(object):
    regions: List[Region]
    edges: List[Tuple[int, int]]

    def __iter__(self) -> Iterator[Tuple[int, int]]:
        # naive default for now is to pass everything twice
        return iter(chain(self.edges, self.edges))

    def __post_init__(self):
        self.into = [list() for _ in self.regions]
        self.outfrom = [list() for _ in self.regions]
        for s, t in self.edges:
            # TODO: ensure that t is a strict subset of s
            self.into[t].append(s)
            self.outfrom[s].append(t)

    def get_regions_with_var(self, variable) -> Iterable[Region]:
        for r in regions:
            for v in r.region_variables:
                if variable.origin.overlaps(v):
                    yield r
                    break  # go to the next region

    @property
    def reachable_from(self, i) -> Iterable[int]:
        yield i
        for t in self.outfrom(i):
            yield self.reachable_from(t)

    @property
    def penetrating_edges(self, i) -> Iterable[Tuple[int, int]]:
        r"""
        returns the set of edges s->t such that s is not reachable from i,
        but t is. (i.e. the set of edges that poke into the region of i)
        """
        return [(s, t) for t in self.reachable_from(i) for s in self.into(t)]


def BetheTree(graph: FactorGraph) -> Strategy:
    # TODO: add in blank factors for queries if necessary
    return Strategy(
        regions=[
            Region((factor_node, *graph.neighbors[factor_node]), 1.0)
            for factor_node in graph.factor_nodes] + [
            Region((variable_node,), 1 - len(graph.neighbors[variable_node]))
            for variable_node in graph.variable_nodes],
        edges=[(i, j) for i in range(graph.num_nodes) for j in graph.neighbors[i]])


class BPInference:
    def __init__(self, graph: FactorGraph, strategy: Strategy):
        self.graph = graph
        self.strategy = strategy
        # the message from one region to another will be a factor dealing with the
        # variables of the target after excluding those of the source
        #
        self.messages: Dict[Tuple[int, int], TensorFactor] = {}
        # these will be the queryf functions
        # self.message_functions: List[Callable[[Sequence[Factor]], Sequence[Factor]]] = []
        # self.update_message_functions: List[Callable[[], None]] = []
        # self.message_outputs: List[List[TensorFactor]] = []
        # self.message_inputs: List[List[Factor]] = []
        # )]

    def logz(self) -> Tensor:
        region_free_energies = []
        for rid, r in enumerate(self.strategy.regions):
            region_free_energies.append(
                r.counting_number * r.free_energy(self.in_messages(rid))
            )
        return -torch.sum(region_free_energies)

    def belief(self, variable: VarBase) -> Tensor:
        r"""
        Each input variable has a tensor and an ndslice (or None to represent a
        request for the estimate of log Z); for each, we will return a
        tensor with one extra dimension; since there may be overlap in the model,
        we will find all regions with the given variable and create a
        final marginal as the average (in log space) of each cell.
        1) find all regions using that variable (we can skip if they don't overlap with the slice of interest)
        2) find belief of that variable according to each region
        3) form a tensor that has the counts
        4) create the average for just the ndslice we care about

        Returns the
        normalized belief corresponding to
        """
        t = torch.zeros(variable.original_tensor.shape + (len(variable.domain),))
        bel = torch.zeros_like(t)
        for region, v in self.strategy.get_regions_with_var(variable):
            t[v.ndslice] += 1
            bel[v.ndslice] += region.query(v)
        return (bel / t)[variable.ndslice]

    def message(self, key: Tuple[int, int]) -> TensorFactor:
        try:
            return self.messages[key]
        except KeyError:
            _, t = key
            return self.messages.setdefault(key, TensorFactor([self.strategy.regions[t].variables]))

    def in_messages(self, region_id):
        pokes_s = self.strategy.penetrating_edges(region_id)
        return [self.message(m) for m in pokes_s]

    @ cache
    def update_messages_from_regionf(self, source_id: int, target_ids: Tuple[int, ...]
                                     ) -> Callable[[], None]:
        source = self.strategy.regions[source_id]
        targets = [self.strategy.regions[target_id] for target_id in target_ids]
        out_messages = [self.messages[source_id, target_id] for target_id in target_ids]
        in_messages = self.in_messages(source_id)
        compute_numerators = source.queryf(in_messages, *[out.variables for out in out_messages])

        pokes_s = self.strategy.penetrating_edges(source_id)
        set_pokes_s = set(pokes_s)
        divide_out_messages = [
            [self.messages[m]
                for m in self.strategy.penetrating_edges(target_id) if m not in set_pokes_s]
            for target_id in target_ids
        ]
        compute_denominators = [target.queryf(terms + [out], source, out.variables)
                                for target, out, terms in zip(targets, divide_out_messages, out_messages)]

        # I want to cache the setup here, but I want it to be flexible??
        def f():
            # compute numerators
            numerators = compute_numerators()
            for numerator, out, compute_denominator, terms in zip(
                    numerators, out_messages, compute_denominators):
                denominator = compute_denominator()
                # - and + rather than / and * since this is in log space
                out.tensor = numerator.tensor - (out.tensor + denominator.tensor)
                out.tensor = DensableFactor.normalize(out.variables, out.tensor)
        return f

    def run(self):
        for s, ts in self.strategy:
            self.update_messages_from_regionf(s, tuple(ts))()


SubjectType = TypeVar('SubjectType', bound='Subject')


ExampleType = TypeVar('ExampleType')


@dataclass
class ListDataset(Dataset, Generic[ExampleType]):
    examples: List[ExampleType]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index) -> ExampleType:
        return self.examples[index]


@dataclass
class Subject:
    is_stacked: bool = field(init=False, default=False)
    __lists: Dict[object, List[object]] = field(init=False, default_factory=dict)
    __vars: FrozenSet = field(init=False, default=frozenset())

    def init_variables(self):
        cls = type(self)
        vars = []
        # TODO: should this just be fields?
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if isinstance(attr, Var):
                property = cast(Var, getattr(self, attr_name))
                if property.tensor is None:
                    raise ValueError(
                        "need to specify an actual tensor for every variable in the subject")
                if property.domain is OPEN_DOMAIN:
                    property._domain = attr.domain
                if property.usage is None:
                    property.usage = torch.full_like(property.tensor, attr.usage.value)
            vars.append(attr)
        self.__vars = frozenset(vars)

    # if this object has been stacked, then:
    # 1) (it will know it and not allow stacking again for now)
    # 2) all variables will be replaced with stacked and padded variables
    # 3) other values will take the value of the first object, but
    #    --- the full list will be accessible via stacked.list(stacked.item)
    #
    @staticmethod
    def stack(subjects: Sequence[SubjectType]) -> SubjectType:
        if not subjects:
            raise ValueError(
                "Your list of subjects needs to have at least one in it")
        first = subjects[0]
        if first.is_stacked:
            raise ValueError(
                "Not allowed to stack already stacked subjects")
        out = copy.deepcopy(first)
        out.is_stacked = True
        cls = type(out)
        my_fields = set(field.name for field in fields(first)) - first.__vars
        for attr_name in first.__vars:
            attr = cast(Var, getattr(out, attr_name))
            stacked = Var.pad_and_stack([
                cast(Var, getattr(subject, attr_name))
                for subject in subjects])
            setattr(out, attr_name, stacked)
        for attr_name in my_fields:
            attr = getattr(out, attr_name)
            out.__lists[attr] = [
                getattr(subject, attr_name)
                for subject in subjects]
        return out

    def data_loader(data: Union[List[ExampleType], Dataset], **kwargs) -> DataLoader:
        if not isinstance(data, Dataset):
            data = ListDataset(data)
        return DataLoader(cast(Dataset, data), collate_fn=Subject.stack, **kwargs)
        # def shapes(self):
        #     cls = type(obj)
        #     for attr_name in dir(cls):
        #         attr = getattr(cls, attr_name)
        #         if isinstance(attr, Var):
        #             property = cast(Var, getattr(obj, attr_name))
        #             if property.tensor is None:
        #                 raise ValueError(
        #                     "need to specify an actual tensor for every variable in the subject")
        #             if property.domain is OPEN_DOMAIN:
        #                 property._domain = attr.domain
        #             if property.usage is None:
        #                 property.usage = torch.full_like(property.tensor, attr.usage.value)

        # @staticmethod
        # def collate(subjects: Sequence[SubjectType]) -> SubjectType:

        #     return

    def clamp_annotated(self) -> None:
        for attr_name in self.__vars:
            cast(Var, getattr(self, attr_name)).clamp_annotated()

    def unclamp_annotated(self) -> None:
        for attr_name in self.__vars:
            cast(Var, getattr(self, attr_name)).unclamp_annotated()

    def __post_init__(self):
        self.init_variables()

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
    __default: ClassVar[Tensor] = torch.tensor(0.0)
    params: ParamNamespace
    input: Tensor = __default
    bias: bool = True
    # input_dimensions: int = 1

    @cached_property
    def in_shape(self):
        return tuple(self.input.shape[len(self.batches_shape):])

    @cached_property
    def in_cells(self):
        return math.prod(self.in_shape)

    def dense_(self) -> Tensor:
        r"""returns a tensor that characterizes this factor;

        the factor's variable-domains dictate the number and
        size of the final dimensions.
        the variables themselves, then, know how many batch
        dimensions there are.
        """
        m = self.params.module(lambda:
                               torch.nn.Linear(
                                   in_features=self.in_cells,
                                   out_features=self.cells,
                                   bias=self.bias))
        input = self.input
        if not input.shape:
            input = input.expand_as((*self.batches_shape, 1))
        else:
            input = input.reshape((*self.batches_shape, -1))
        return m(input).reshape(self.shape)
