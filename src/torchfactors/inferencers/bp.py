from __future__ import annotations

from functools import lru_cache
from typing import Callable, Dict, List, Sequence, Tuple, Union

import torch
from torch.functional import Tensor

from ..components.tensor_factor import TensorFactor
from ..factor import Factor
from ..factor_graph import FactorGraph
from ..inferencer import Inferencer
from ..strategies.bethe_graph import BetheGraph
from ..strategy import Strategy
from ..variable import Var

cache = lru_cache(maxsize=None)


class BPInference:
    def __init__(self, graph: FactorGraph, strategy: Strategy):
        self.graph = graph
        self.strategy = strategy
        # the message from one region to another will be a factor dealing with the
        # variables of the target after excluding those of the source
        #
        self.messages: Dict[Tuple[int, int], TensorFactor] = {}
        self.messages_changes: Dict[TensorFactor, Tensor] = {}
        # these will be the queryf functions
        # self.message_functions: List[Callable[[Sequence[Factor]], Sequence[Factor]]] = []
        # self.update_message_functions: List[Callable[[], None]] = []
        # self.message_outputs: List[List[TensorFactor]] = []
        # self.message_inputs: List[List[Factor]] = []
        # )]

    def amount_changed(self, old: Tensor, new: Tensor) -> Tensor:
        return torch.nn.functional.kl_div(old.nan_to_num(), new.nan_to_num(),
                                          log_target=True, reduction='batchmean')

    def total_amount_changed(self):
        return torch.stack(tuple(self.messages_changes.values()), 0).sum(dim=0)

    def logz(self) -> torch.Tensor:
        region_free_energies = []
        for rid, r in enumerate(self.strategy.regions):
            region_free_energies.append(
                r.counting_number * r.free_energy(self.in_messages(rid))
            )
        return -torch.stack(region_free_energies, -1).sum(dim=-1)

    def belief(self, variables: Union[Var, Sequence[Var]]) -> torch.Tensor:
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
        # TODO: this could be a lot more efficient
        t = torch.zeros(variable.origin.shape + (len(variable.domain),))
        full_belief = torch.zeros_like(t)
        for region_id, region, vs in self.strategy.get_regions_with_vars(variable):
            region_beliefs = region.product_marginals([(v,) for v in vs],
                                                      other_factors=self.in_messages(region_id))
            for v, region_belief in zip(vs, region_beliefs):
                t[v.out_slice] += 1
                full_belief[v.out_slice] += region_belief
        out = (full_belief / t)[variable.out_slice]
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

        # pokes_s = self.strategy.penetrating_edges(source_id)
        # set_pokes_s = set(pokes_s)
        # we will divide out everything but the message of interest
        divide_out_messages = [
            tuple([self.message(m)
                   for m in self.strategy.penetrating_edges(target_id)
                   if m != (source_id, target_id)])
            for target_id in target_ids]
        compute_denominators = [
            target_region.marginals_closure([out_message.variables], other_factors=denom_messages)
            # exclude=source)
            for target_region, denom_messages, out_message
            in zip(targets, divide_out_messages, out_messages)]

        # I want to cache the setup here, but I want it to be flexible??
        def f():
            # compute numerators
            numerators = compute_numerators()
            for numerator, out, compute_denominator in zip(
                    numerators, out_messages, compute_denominators):
                # keep the nans, but the negative infs can be ignored
                denominator = compute_denominator()[0].nan_to_num(
                    nan=float('nan'), posinf=float('inf'), neginf=0)
                # - and + rather than / and * since this is in log space
                updated = Factor.normalize(numerator - denominator,
                                           num_batch_dims=out.num_batch_dims)
                change = -self.amount_changed(out.tensor, updated)
                self.messages_changes[out] = change
                out.tensor = updated
        return f

    def run(self):
        for s, ts in self.strategy:
            self.update_messages_from_regionf(s, tuple(ts))()


class BP(Inferencer):

    def __init__(self, strategy: Callable[[FactorGraph, int], Strategy] = BetheGraph,
                 passes: int = 3):
        self.strategy_factory = strategy
        self.passes = passes

    # TODO: handle queries that are not in the graph
    def product_marginals_(self, factors: Sequence[Factor], *queries: Sequence[Var],
                           normalize: bool = True, append_total_change: bool = False
                           ) -> Sequence[torch.Tensor]:
        fg = FactorGraph(factors)
        strategy = self.strategy_factory(fg, self.passes)
        bp = BPInference(fg, strategy)
        bp.run()
        if () in queries or not normalize:
            logz = bp.logz()
        responses: List[torch.Tensor] = []
        for query in queries:
            if query == ():
                responses.append(logz)
            else:
                belief = bp.belief(query)
                if not normalize:
                    belief = belief + logz
                responses.append(belief)
        if append_total_change:
            responses.append(bp.total_amount_changed())
        return responses
