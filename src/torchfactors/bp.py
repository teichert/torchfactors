from __future__ import annotations

from typing import (Callable, Dict, Tuple)

import torch
from torch import Tensor


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
