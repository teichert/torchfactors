from __future__ import annotations

from functools import lru_cache
from typing import Callable, Dict, List, Sequence, Tuple, Union

import torch
from torch import Tensor

from torchfactors.factor import Factor, check_queries

from .components.tensor_factor import TensorFactor
from .factor_graph import FactorGraph
from .strategies.bethe_graph import BetheGraph
from .strategy import Strategy
from .variable import Var

cache = lru_cache(maxsize=None)


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
        return -torch.stack(region_free_energies, -1).sum(dim=-1)

    def belief(self, variables: Union[Var, Sequence[Var]]) -> Tensor:
        r"""
        Each input variable has a tensor and an ndslice (or None to represent a
        request for the estimate of log Z); for each, we will return a tensor
        with one extra dimension; since there may be overlap in the model, we
        will find all regions with the given variable and create a final
        marginal as the average (in log space) of each cell.
        1) find all regions using that variable (we can skip if they don't
           overlap with the slice of interest)
        2) find belief of that variable according to each region
        3) form a tensor that has the counts
        4) create the average for just the ndslice we care about

        Returns the normalized belief corresponding to
        """
        # if isinstance(variables, Var):
        #     variables = (variables,)
        assert not isinstance(variables, Var)
        if len(variables) != 1:
            raise ValueError("not ready to handle multi-variable belief queries")
        variable = variables[0]
        t = torch.zeros(variable.original_tensor.shape + (len(variable.domain),))
        full_belief = torch.zeros_like(t)
        for region_id, region, vs in self.strategy.get_regions_with_vars(variable):
            region_beliefs = region.product_marginals([(v,) for v in vs],
                                                      other_factors=self.in_messages(region_id))[0]
            for v, region_belief in zip(vs, region_beliefs):
                t[v.ndslice] += 1
                full_belief[v.ndslice] += region_belief
        out = (full_belief / t)[variable.ndslice]
        return Factor.normalize(out, len(variable.tensor.shape))

    def message(self, key: Tuple[int, int]) -> TensorFactor:
        r"""
        retrieve and return the message between a directed pair of regions (or
        initialize one if there isn't one yet)
        """
        try:
            return self.messages[key]
        except KeyError:
            _, t = key
            return self.messages.setdefault(
                key, TensorFactor(
                    *self.strategy.regions[t].variables,
                    init=torch.zeros))

    def in_messages(self, region_id):
        """
        return a list of all of the message that penetrate the region indicated
        by the given region_id
        """
        pokes_s = self.strategy.penetrating_edges(region_id)
        return [self.message(m) for m in pokes_s]

    @ cache
    def update_messages_from_regionf(self, source_id: int, target_ids: Tuple[int, ...]
                                     ) -> Callable[[], None]:
        r"""
        returns a method that will update all of the messages from the specified
        source to each each of the specified targets
        """
        source = self.strategy.regions[source_id]
        targets = [self.strategy.regions[target_id] for target_id in target_ids]
        out_messages = [self.message((source_id, target_id)) for target_id in target_ids]
        in_messages = self.in_messages(source_id)
        compute_numerators = source.marginals_closure([out.variables for out in out_messages],
                                                      other_factors=in_messages)

        pokes_s = self.strategy.penetrating_edges(source_id)
        set_pokes_s = set(pokes_s)
        divide_out_messages = [
            tuple([self.message(m)
                   for m in self.strategy.penetrating_edges(target_id) if m not in set_pokes_s])
            for target_id in target_ids
        ]
        compute_denominators = [
            target_region.marginals_closure([out_message.variables], other_factors=denom_messages,
                                            exclude=source)
            for target_region, denom_messages, out_message
            in zip(targets, divide_out_messages, out_messages)]

        # I want to cache the setup here, but I want it to be flexible??
        def f():
            # compute numerators
            numerators = compute_numerators()
            for numerator, out, compute_denominator in zip(
                    numerators, out_messages, compute_denominators):
                denominator = compute_denominator()[0]
                # - and + rather than / and * since this is in log space
                out.tensor.data = numerator - (out.tensor + denominator)
                out.tensor.data = Factor.normalize(out.tensor, num_batch_dims=out.num_batch_dims)
        return f

    def run(self):
        for s, ts in self.strategy:
            self.update_messages_from_regionf(s, tuple(ts))()


def product_marginal(factor_graph: FactorGraph, query: Union[Sequence[Var], Var, None] = None,
                     strategy: Strategy = None, normalize=True
                     ) -> Tensor:
    r"""
    convenience method for a single product_marginal query
    """
    if query is None:
        query = ()
    elif isinstance(query, Var):
        query = (query,)
    out, = product_marginals(factor_graph,
                             query, strategy=strategy,
                             force_multi=True, normalize=normalize)
    return out


# TODO: handle queries that are not in the graph
def product_marginals(factor_graph: FactorGraph, *queries: Sequence[Var],
                      strategy: Strategy = None, force_multi=False, normalize=True
                      ) -> Union[Sequence[Tensor], Tensor]:
    r"""
    Returns marginals corresponding to the specified queries.
    These Given a factor_graph and a strategy (default it bethe-graph).
    If `normalize` is specified as True (note, this is usually more efficient
    than not normalizing), then each marginal tensor will logsumexp to 0
    with the exception of a query for () which is used to represent a request
    for the log partition function. If normalize is False,
    then the partition function will be used to unnormalize
    the normalized belief.
    The idea is that the normalized belief came from dividing the unnormalized by Z,
    so I get the unnormalized back by multiplying by Z:
    b = m / z => m = b * z
    """
    check_queries(queries)
    if strategy is None:
        strategy = BetheGraph(factor_graph)
    # query_list = [(q,) if isinstance(q, VarBase) else q for q in queries]
    bp = BPInference(factor_graph, strategy)
    bp.run()
    if () in queries or not normalize:
        logz = bp.logz()
    responses: List[Tensor] = []
    for query in queries:
        if query == ():
            responses.append(logz)
        else:
            belief = bp.belief(query)
            if not normalize:
                belief = belief + logz
            responses.append(belief)
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
