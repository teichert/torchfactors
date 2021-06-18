from __future__ import annotations

from typing import (Iterator, List, Optional, Sequence, Union)

from torch import Tensor


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
