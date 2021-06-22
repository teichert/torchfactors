from __future__ import annotations

from typing import Iterator, List, Sequence

from .factor import Factor
from .variable import Var


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
                         ) -> List[Var]:
        return [
            self.variables[node_id - self.num_factors]
            for node_id in self.full_region(node_ids)
        ]
