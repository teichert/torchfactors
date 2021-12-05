from __future__ import annotations

import math
# from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union, cast

import torch
from torch.functional import Tensor
from torchmetrics.functional import kl_divergence
from tqdm import tqdm  # type: ignore

from ..components.tensor_factor import Message, TensorFactor
from ..factor import Factor
from ..factor_graph import FactorGraph
from ..inferencer import Inferencer
from ..strategies.bethe_graph import BetheGraph
from ..strategy import Strategy
from ..utils import sum_tensors
from ..variable import TensorVar, Var, at

# cache = lru_cache(maxsize=None)


class BPInference:
    r"""
    Generalized belief propagation (sum-product) inference (in log space).

    The strategy specifies the graph-based approximation which can be
    exact if a junction-tree based strategy is used.
    """

    def __init__(self, graph: FactorGraph, strategy: Strategy,
                 compute_change: bool = False):
        self.graph = graph
        self.strategy = strategy
        # the message from one region to another will be a factor dealing with the
        # variables of the target
        self.messages: Dict[Tuple[int, int], TensorFactor] = {}
        self.total_amount_changed: Optional[Tensor] = None
        self.compute_change = compute_change

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
        out = kl_divergence(new, Factor.normalize(
            old, num_batch_dims=num_batch_dims), log_prob=True)
        return out

    def logz(self) -> torch.Tensor:
        region_free_energies = []
        for rid, r in enumerate(tqdm(self.strategy.regions, desc="Computing region energies...",
                                     delay=0.5, leave=False)):
            this_free_energy = r.free_energy(self.in_messages(rid))
            if r.ndims > 0:
                end = len(this_free_energy.shape)
                start = end - r.ndims
                this_free_energy = this_free_energy.sum(dim=list(range(start, end)))
            region_free_energies.append(r.counting_number * this_free_energy)
        return -sum_tensors(region_free_energies)

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

    def update_messages_from_region(self, source_id: int, target_ids: Tuple[int, ...],
                                    accumulate_change=False
                                    ):  # -> Callable[[], None]:
        r"""
        returns a method that will update all of the messages from the specified
        source to each each of the specified targets
        """
        source = self.strategy.regions[source_id]
        out_messages = [self.message((source_id, target_id)) for target_id in target_ids]
        in_messages = self.in_messages(source_id)
        numerators = source.product_marginals([out.variables for out in out_messages],
                                              other_factors=in_messages)
        updated_messages = []
        for target_id, numerator, out in zip(
                target_ids, numerators, out_messages):
            target = self.strategy.regions[target_id]
            messages_to_divide_out = tuple([self.message(m)
                                            for m in self.strategy.penetrating_edges(target_id)
                                            if m != (source_id, target_id)])
            denominator, = target.product_marginals([out.variables],
                                                    other_factors=messages_to_divide_out)
            denominator = denominator.masked_fill(
                denominator == float('-inf'), 0)
            # - and + rather than / and * since this is in log space
            updated_messages.append(Factor.normalize(numerator - denominator,
                                                     num_batch_dims=out.num_batch_dims))
        for out, updated in zip(out_messages, updated_messages):
            # keep track of how far from convergence we were before this message
            if accumulate_change:
                change = self.amount_changed(out.tensor, updated, out.num_batch_dims).sum()
                if self.total_amount_changed is None:
                    self.total_amount_changed = change
                else:
                    self.total_amount_changed = self.total_amount_changed + change
            else:  # not changing on penalty pass
                out.tensor = updated

    def run(self):
        for s, ts in tqdm(list(self.strategy), leave=False, delay=1, desc='Passing messages...'):
            self.update_messages_from_region(s, tuple(ts))
        if self.compute_change:
            self.total_amount_changed = None
            for s, ts in tqdm(self.strategy.edge_groups(), leave=False, delay=1,
                              desc='Computing change...'):
                self.update_messages_from_region(s, tuple(ts), accumulate_change=True)

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
        bp = BPInference(fg, strategy, compute_change=append_total_change)
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
            responses.append(cast(Tensor, bp.total_amount_changed))
        return responses
