from __future__ import annotations

import math
from functools import lru_cache
from typing import Callable, Dict, List, Sequence, Tuple, Union

import torch
from torch.functional import Tensor
from torchmetrics.functional import kldivergence
from tqdm import tqdm  # type: ignore

from ..components.tensor_factor import Message, TensorFactor
from ..factor import Factor
from ..factor_graph import FactorGraph
from ..inferencer import Inferencer
from ..strategies.bethe_graph import BetheGraph
from ..strategy import Strategy
from ..variable import TensorVar, Var, at

cache = lru_cache(maxsize=None)


class BPInference:
    r"""
    Generalized belief propagation (sum-product) inference (in log space).

    The strategy specifies the graph-based approximation which can be
    exact if a junction-tree based strategy is used.
    """

    def __init__(self, graph: FactorGraph, strategy: Strategy):
        self.graph = graph
        self.strategy = strategy
        # the message from one region to another will be a factor dealing with the
        # variables of the target
        self.messages: Dict[Tuple[int, int], TensorFactor] = {}
        self.message_changes: Dict[TensorFactor, Tensor] = {}

    def amount_changed(self, old: Tensor, new: Tensor, num_batch_dims: int) -> Tensor:
        num_dims = len(old.shape)
        old = old.nan_to_num()
        new = new.nan_to_num()
        if num_dims < 2:
            old = old[(None,)*(2 - num_dims)]
            new = new[(None,)*(2 - num_dims)]
            num_batch_dims = 1
        num_dims = len(old.shape)
        old = old.flatten(0, num_batch_dims - 1).flatten(1)
        new = new.flatten(0, num_batch_dims - 1).flatten(1)
        out = kldivergence(new, Factor.normalize(
            old, num_batch_dims=num_batch_dims), log_prob=True)
        return out

    def total_amount_changed(self):
        r"""returns the sum of the KL divergences from the most recent message
        to the previous version.  If this value is 0, then convergence has been
        achieved."""
        changes = torch.tensor(list(self.message_changes.values())).sum()
        return changes

    def logz(self) -> torch.Tensor:
        region_free_energies = []
        for rid, r in enumerate(self.strategy.regions):
            this_free_energy = r.free_energy(self.in_messages(rid))
            region_free_energies.append(r.counting_number * this_free_energy)
        return -torch.stack(region_free_energies, -1).sum(dim=-1)

    def region_belief(self, variable: Var) -> Tensor:
        if variable in self.strategy.var_to_region:
            region_id = self.strategy.var_to_region[variable]
            region = self.strategy.regions[region_id]
            region_beliefs, = region.product_marginals(
                [[variable]], other_factors=self.in_messages(region_id))
            return Factor.normalize(region_beliefs, len(variable.tensor.shape))
        else:
            # TODO: there is some waste here
            full = torch.zeros(
                variable.origin.marginal_shape).as_subclass(torch.Tensor)  # type: ignore
            for sub_var in self.strategy.root_to_subs[variable.origin]:
                sub_belief = self.region_belief(sub_var)
                at(full, sub_var.out_slice)[(...,)] = sub_belief
            return full

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
        assert not isinstance(variables, Var)
        # TODO: how to handle overaping variables?
        if len(variables) != 1:
            raise ValueError("not ready to handle multi-variable belief queries")
        variable = variables[0]
        return self.region_belief(variable)

    def message(self, key: Tuple[int, int]) -> TensorFactor:
        r"""
        retrieve and return the message between a directed pair of regions (or
        initialize one if there isn't one yet)
        """
        try:
            return self.messages[key]
        except KeyError:
            _, t = key
            v: Var = self.strategy.regions[t].variables[0]
            variable_shape = v.shape

            # make a uniform message backed by a single scalar
            def init(shape):
                num_configs = math.prod(shape[len(variable_shape):])
                out = (-torch.tensor(num_configs).float().log()).expand(shape)
                return out

            message = Message(
                *self.strategy.regions[t].variables,
                init=init)

            return self.messages.setdefault(key, message)

    def in_messages(self, region_id: int) -> Sequence[TensorFactor]:
        """
        return a list of all of the message that penetrate the region indicated
        by the given region_id
        """
        pokes_s = self.strategy.penetrating_edges(region_id)
        return [self.message(m) for m in pokes_s]

    # @ cache
    def update_messages_from_region(self, source_id: int, target_ids: Tuple[int, ...]
                                    ):  # -> Callable[[], None]:
        r"""
        returns a method that will update all of the messages from the specified
        source to each each of the specified targets
        """
        source = self.strategy.regions[source_id]
        targets = [self.strategy.regions[target_id] for target_id in target_ids]
        out_messages = [self.message((source_id, target_id)) for target_id in target_ids]
        in_messages = self.in_messages(source_id)
        numerators = source.product_marginals([out.variables for out in out_messages],
                                              other_factors=in_messages)

        # we will divide out everything but the message of interest
        messages_to_divide_out_per_target = [
            tuple([self.message(m)
                   for m in self.strategy.penetrating_edges(target_id)
                   if m != (source_id, target_id)])
            for target_id in target_ids]
        denominators = [
            target_region.product_marginals([out_message.variables], other_factors=denom_messages)
            for target_region, denom_messages, out_message
            in zip(targets, messages_to_divide_out_per_target, out_messages)]

        # I want to cache the setup here, but I want it to be flexible??
        # def update_messages():
        # numerators = compute_numerators()
        for numerator, out, denominator in zip(
                numerators, out_messages, denominators):
            # keep the nans, but the negative infs can be ignored
            denominator = denominator[0].nan_to_num(
                nan=float('nan'), posinf=float('inf'), neginf=0)
            # - and + rather than / and * since this is in log space
            updated = Factor.normalize(numerator - denominator,
                                       num_batch_dims=out.num_batch_dims)
            # keep track of how far from convergence we were before this message
            change = self.amount_changed(out.tensor, updated, out.num_batch_dims)
            self.message_changes[out] = change
            out.tensor = updated
        # return update_messages

    def run(self):
        for s, ts in tqdm(list(self.strategy), leave=False, delay=1):
            self.update_messages_from_region(s, tuple(ts))

    def display(self):  # pragma: no cover
        """display the variables and factors for debugging"""
        def var_name(v: Var) -> str:
            if isinstance(v, TensorVar):
                return str(v._info)
            else:
                return str(f'v{id(v)}')
        fg = self.graph
        fg.variable_nodes
        variables = sorted(set((var_name(v), vid, v) for v, vid in fg.varids.items()))
        factors = sorted(set(
            (tuple(var_name(v) for v in f.variables), fid, f)
            for fid, f in enumerate(fg.factors)))
        print("Variable: Node-Id")
        for name, vid, v in variables:
            print(f'{name}:\t{vid}, {v}')

        print('Factors: Node-Id')
        for vids, fid, f in factors:
            name = ','.join(str(v) for v in vids)
            print(f'[{name}]:\t{fid}, {f.dense.tolist()}, {f}')

        print('Regions (variables; factors): Region-Id')
        region_names = []
        for rid, region in enumerate(self.strategy.regions):
            region_vars = tuple(sorted(tuple(var_name(v) for v in region.variables))),
            region_factors = tuple(tuple(var_name(v) for v in f.variables)
                                   for f in region.factors),
            region = self.strategy.regions[rid]
            var_names = ','.join(str(v) for v in region_vars)
            factor_names = '],['.join(','.join(str(v) for v in vids) for vids in region_factors)
            region_name = f'({var_names}; [{factor_names}]):\t{rid}'
            region_names.append(region_name)
            belief = Factor.normalize(region.product_marginals(
                [region.variables], other_factors=self.in_messages(rid))[0])
            print(f'{region_name}, {belief.tolist()}, {region}')

        print('Messages: (source->target)')
        for target, source in sorted((target, source) for source, target in self.messages.keys()):
            message = self.messages[source, target]
            print(
                f'{region_names[source]}->{region_names[target]}, '
                f'{message.dense.tolist()}, {message}')


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
