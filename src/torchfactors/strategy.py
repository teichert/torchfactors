from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
from typing import (DefaultDict, Dict, FrozenSet, Iterable, Iterator, List,
                    Sequence, Set, Tuple)

import torch
from torch import Tensor

from .components.tensor_factor import TensorFactor
from .factor import Factor
from .factor_graph import FactorGraph
from .variable import Var


def pick_controller(factors: Iterable[Factor]) -> Tuple[Factor, Sequence[Factor]]:
    r"""
    isolates the factor with the most variables
    """
    all_factors = list(factors)
    _, ix, controller = max((len(f.variables), i, f)
                            for i, f in enumerate(all_factors))
    other_factors = list(all_factors[:ix]) + list(all_factors[(ix + 1):])
    return controller, other_factors


@dataclass
class Region(object):
    factor_graph: FactorGraph
    factor_graph_nodes: Tuple[int, ...]
    counting_number: float

    @property
    def ndims(self) -> int:
        return self.factor_graph.region_variables(self.factor_graph_nodes)[0].ndims

    @property
    def variables(self) -> Sequence[Var]:
        return self.factor_graph.region_variables(self.factor_graph_nodes)

    @property
    def factors(self) -> Sequence[Factor]:
        return self.factor_graph.region_factors(self.factor_graph_nodes)

    @property
    def factor_set(self) -> FrozenSet[Factor]:
        return frozenset(self.factors)

    def product_marginals(self,
                          queries: Sequence[Sequence[Var]] = (()),
                          other_factors: Sequence[Factor] = (()),
                          ):
        r"""
        analogous to the function with the same name on the Factor class except with a slightly
        different interface (to avoid needing to special-case or risk errors) and with the
        ability to exclude the factors from a particular region
        """
        # factors appearing in the excluded region are excluded; note that the
        # first factor (in the region) touching the most variables determines
        # how the inference is done (if this needs to be modified, then have
        # your strategy create subclasses of region that override this behavior)
        # surviving_factors = list((self.factor_set - exclude.factor_set)
        #                          if exclude is not None else self.factor_set)
        if not self.factors and not other_factors:
            return [TensorFactor(*query).dense for query in queries]
        controller, others = pick_controller(chain(self.factors, other_factors))
        return controller.product_marginals(*queries, other_factors=others)

    def free_energy(self, messages: Sequence[Factor]) -> Tensor:
        """
        analogous to the function with the same name on the Factor class; the
        messages are included in the product when computing the belief; no other
        energy functions are allowed; uses the factor within the region that has the
        largest number of participating variables
        """
        factors = self.factors
        if not factors:
            factors = [TensorFactor(*self.variables, init=torch.zeros)]
        controller, others = pick_controller(factors)
        return controller.free_energy(other_energy=others,
                                      messages=messages)


@dataclass
class Strategy(object):
    r"""
    Represents a strategy for approximate (or exact) inference on a factor
    graph. Specifically, it is a region-graph approximation: Regions of the
    factor graph are specified and directed edges from super regions to sub
    regions are given.  Each region is made up of 0 or more factors (and all the
    varibles they touch) and 0 or more additional variables.  Each region also
    specified a counting number which should be such that each node in the graph
    is counted exactly once.
    """
    regions: List[Region]
    edges: List[Tuple[int, int]]
    passes: int = 2

    def edge_groups(self) -> List[Tuple[int, List[int]]]:
        return list((s, ts) for s, ts in enumerate(self.outfrom) if ts)

    def __iter__(self) -> Iterator[Tuple[int, List[int]]]:
        # naive default for now is to pass everything a fixed number of times
        schedule = self.edge_groups()
        return itertools.chain.from_iterable(itertools.repeat(schedule, self.passes))

    def __post_init__(self):
        self.into: List[List[int]] = [list() for _ in self.regions]
        self.outfrom: List[List[int]] = [list() for _ in self.regions]
        # maps from a variable to the smallest region holding it
        # (not reasoning about super/sub variables;
        var_to_size_and_region: Dict[Var, Tuple[int, int]] = {}
        self.root_to_subs: DefaultDict[Var, Set[Var]] = defaultdict(set)
        for rid, r in enumerate(self.regions):
            for v in r.variables:
                if v.origin is not v:
                    self.root_to_subs[v.origin].add(v)
                this_one = (len(r.variables), rid)
                if v not in var_to_size_and_region:
                    var_to_size_and_region[v] = this_one
                else:
                    old_one = var_to_size_and_region[v]
                    if this_one < old_one:
                        var_to_size_and_region[v] = this_one
        self.var_to_region: Dict[Var, int] = {v: region_id for v,
                                              (_, region_id) in var_to_size_and_region.items()}
        for s, t in self.edges:
            self.into[t].append(s)
            self.outfrom[s].append(t)

    def reachable_from(self, i: int) -> Iterable[int]:
        r"""
        Generates a list of all of the region ids that are reachable from the
        region id `i` via the directed edges in the region graph (`i` is always
        included)
        """
        yield i
        for t in self.outfrom[i]:
            yield from self.reachable_from(t)

    def penetrating_edges(self, i: int) -> Iterable[Tuple[int, int]]:
        r"""
        returns the set of edges s->t such that s is not reachable from i,
        but t is. (i.e. the set of edges that poke into the region of i)
        """
        from_i = set(self.reachable_from(i))
        return [(s, t) for t in from_i for s in self.into[t] if s not in from_i]
