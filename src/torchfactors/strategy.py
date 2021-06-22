from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, lru_cache
from itertools import chain
from typing import (Callable, FrozenSet, Iterable, Iterator, List, Optional,
                    Sequence, Tuple)

from torch import Tensor

from .factor import Factor
from .factor_graph import FactorGraph
from .variable import Var

cache = lru_cache(maxsize=None)


@ dataclass
class Region(object):
    factor_graph: FactorGraph
    factor_graph_nodes: Tuple[int, ...]
    counting_number: float

    @ cached_property
    def variables(self) -> Sequence[Var]:
        return self.factor_graph.region_variables(self.factor_graph_nodes)

    @ cached_property
    def factors(self) -> Sequence[Factor]:
        return self.factor_graph.region_factors(self.factor_graph_nodes)

    @ cached_property
    def factor_set(self) -> FrozenSet[Factor]:
        return frozenset(self.factors)

    def query(self, others: Sequence[Factor],
              *queries: Sequence[Var], exclude: Optional['Region'] = None):
        return self.queryf(others, *queries, exclude=exclude)

    @ cache
    def queryf(self, others: Sequence[Factor], exclude: Optional['Region'],
               *queries: Sequence[Var]
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
    def free_energy(self, messages: Sequence['Factor']) -> Tensor:
        # _, ix, controller = max((len(f.variables), i, f) for i, f in enumerate(self.factors))
        pass


@ dataclass
class Strategy(object):
    regions: List[Region]
    edges: List[Tuple[int, int]]

    def __iter__(self) -> Iterator[Tuple[int, Tuple[int, ...]]]:
        # naive default for now is to pass everything twice
        schedule = [(s, (t,)) for s, t in self.edges]
        return iter(chain(schedule, schedule))

    def __post_init__(self):
        self.into: List[List[int]] = [list() for _ in self.regions]
        self.outfrom: List[List[int]] = [list() for _ in self.regions]
        for s, t in self.edges:
            # TODO: ensure that t is a strict subset of s
            self.into[t].append(s)
            self.outfrom[s].append(t)

    def get_regions_with_var(self, variable) -> Iterable[Tuple[int, Region, Var]]:
        for rid, r in enumerate(self.regions):
            for v in r.variables:
                if variable.origin.overlaps(v):
                    yield rid, r, v
                    break  # go to the next region

    def reachable_from(self, i) -> Iterable[int]:
        yield i
        for t in self.outfrom[i]:
            yield from self.reachable_from(t)

    def penetrating_edges(self, i) -> Iterable[Tuple[int, int]]:
        r"""
        returns the set of edges s->t such that s is not reachable from i,
        but t is. (i.e. the set of edges that poke into the region of i)
        """
        return [(s, t) for t in self.reachable_from(i) for s in self.into[t]]
