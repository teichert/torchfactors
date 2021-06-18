from __future__ import annotations

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


@ dataclass
class Region(object):
    factor_graph: FactorGraph
    factor_graph_nodes: Tuple[int, ...]
    counting_number: float

    @ cached_property
    def variables(self) -> Sequence[VarBase]:
        return self.factor_graph.region_variables(self.factor_graph_nodes)

    @ cached_property
    def factors(self) -> Sequence[Factor]:
        return self.factor_graph.region_factors(self.factor_graph_nodes)

    @ cached_property
    def factor_set(self) -> FrozenSet[Factor]:
        return frozenset(self.factors)

    def query(self, others: Sequence[Factor],
              *queries: Sequence[VarBase], exclude: Optional['Region'] = None):
        return self.queryf(others, *queries, exclude=exclude)

    @ cache
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


@ dataclass
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

    @ property
    def reachable_from(self, i) -> Iterable[int]:
        yield i
        for t in self.outfrom(i):
            yield self.reachable_from(t)

    @ property
    def penetrating_edges(self, i) -> Iterable[Tuple[int, int]]:
        r"""
        returns the set of edges s->t such that s is not reachable from i,
        but t is. (i.e. the set of edges that poke into the region of i)
        """
        return [(s, t) for t in self.reachable_from(i) for s in self.into(t)]
