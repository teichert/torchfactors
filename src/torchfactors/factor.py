from __future__ import annotations

import math
from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Iterator, List, Sequence, cast

import torch
from torch import Tensor

from torchfactors import einsum

from .variable import Var


@dataclass
class Factor:
    r"""
    A Factor has a domain which is defined by the set of variables it is
    concerned with. It is a function from any configuration of those variables
    to a mass value (i.e. we only consider non-negative values which is why we
    can operate in the log space).  To support inference among many factors, a
    factor need to know how to answer queries given other "denseable" factors
    as input and a set of (einsum style) queries to respond to.
    """
    variables: Sequence[Var]

    def __iter__(self) -> Iterator[Var]:
        return iter(self.variables)

    def __len__(self) -> int:
        return len(self.variables)

    def query(self, others: Sequence[Factor], *queries: Sequence[Var]
              ) -> Sequence[Tensor]:
        return self.queryf([f.variables for f in others], *queries)(
            others)

    # @abstractmethod
    def queryf(self, others: Sequence[Sequence[Var]], *queries: Sequence[Var]
               ) -> Callable[[Sequence[Factor]], Sequence[Tensor]]:
        raise NotImplementedError("don't know how to do queries on this")

    def free_energy(self, other_energy: Sequence[Factor], messages: Sequence[Factor]
                    ) -> Tensor:
        """
        an estimate of the contribution of this factor to the -log z;
        it is the entropy minus the average energy
        under a distribution given by the normalized product of all energy
        factors and all messages
        """
        raise NotImplementedError("don't know how to do queries on this")

    def dense(self) -> Tensor:
        raise NotImplementedError("don't know how to give a dense version of this")

    @cached_property
    def out_shape(self):
        return tuple([len(t.domain) for t in self.variables])

    @cached_property
    def shape(self):
        return tuple(*self.batches_shape, *self.out_shape)

    @cached_property
    def cells(self):
        return math.prod(self.shape)

    @cached_property
    def batch_cells(self):
        return math.prod(self.batches_shape)

    @cached_property
    def batches_shape(self):
        # should be the same for all variables (maybe worth checking?)
        first = self.variables[0]
        return tuple(first.tensor.shape[:-1])

    @cached_property
    def num_batch_dims(self):
        return len(self.batches_shape)

    # @property
    # def excluded_mask(self):


# def inverse


# class Factors(ABC, Iterable[Factor]):
#     r"""
#     A collection of factors (sometimes a coherent model component is most easily
#     described as a subclass of this)
#     """

#     def __iter__(self) -> Iterator[Factor]:
#         return self.factors()

#     @abstractmethod
#     def factors(self) -> Iterator[Factor]:
#         pass


class DensableFactor(Factor):

    @abstractmethod
    def dense_(self) -> Tensor: ...

    @property
    def dense(self) -> Tensor:
        return self.dense_()
        # I only care about fixing the output here (don't care about observed
        # inputs since those have already been clamped and set to nan)
        # excluded_mask is anything that is clamped or observed and not the
        # current value as well as anything that is padded and not 0 TODO:
        # finish this I have a tensor where the value at position x,y is the
        # index of the z coordinate that I want

        # import math
        # inp = torch.tensor([
        #     [  # b0
        #         2,  # i0
        #         3,  # i2
        #     ],
        #     [  # b1
        #         3,  # i0
        #         2,  # i2
        #     ],
        # ])

        # dims = (*inp.shape, 4)
        # t = torch.arange(math.prod(dims)).reshape(dims)
        # # t.reshape(-1, inp.numel())[range(inp.numel()), inp.reshape(inp.numel())]
        # mask = torch.ones_like(t).bool()
        # mask.reshape(-1, inp.numel())[range(inp.numel()), inp.reshape(inp.numel())] = 0

        # t[mask]
        # input[batch_row, batch_col, index]: (v: 5)
        # [ # br0
        #     [ # bc0
        #         0,  # i0
        #         1,  # 11
        #         2,  # i2
        #     ],
        #     [ # bc1
        #         1,  # i0
        #         2,  # 11
        #         3,  # i2
        #     ],
        #     [ # bc2
        #         2,  # i0
        #         3,  # 11
        #         4,  # i2
        #     ],
        # ],
        # [ # br1
        #     [ # bc0
        #         0,  # i0
        #         1,  # 11
        #         2,  # i2
        #     ],
        #     [ # bc1
        #         1,  # i0
        #         2,  # 11
        #         3,  # i2
        #     ],
        #     [ # bc2
        #         2,  # i0
        #         3,  # 11
        #         4,  # i2
        #     ],
        # ],
        # [ # br2
        #     [ # bc0
        #         0,  # i0
        #         1,  # 11
        #         2,  # i2
        #     ],
        #     [ # bc1
        #         1,  # i0
        #         2,  # 11
        #         3,  # i2
        #     ],
        #     [ # bc2
        #         2,  # i0
        #         3,  # 11
        #         4,  # i2
        #     ],
        # ],

        # for i, v in enumerate(reversed(self.variables)):
        #     mask = torch.ones_like(v.tensor, dtype=bool)
        #     # only keep the values represented in the current tensor
        #     # for observed, clamped, or padding (not for annotated or latent)
        #     v.usage_is_fixed
        #     # TODO: merge in above
        #     torch.logical_or(
        #         (v.usage == VarUsage.OBSERVED),
        #         torch.logical_or(
        #             (v.usage == VarUsage.CLAMPED),
        #             (v.usage == VarUsage.PADDING)))

        #      v.usage == VarUsage.PADDING).logical_or(
        #     (d.movedim(self.num_batch_dims, self.num_batch_dims + i)
        #       [[((
        #             v.usage == VarUsage.OBSERVED).logical_or(
        #             v.usage == VarUsage.CLAMPED).logical_or(
        #             v.usage == VarUsage.CLAMPED).logical_or(
        #         v.usage == VarUsage.PADDING) > 0]+[...]]=fl

        # d[self.excluded_mask]=float('-inf')
        # # clamped_mask is anything that is clamped or observed and is the target
        # d[self.clamped_mask]=0.0
        # # padded_mask is anything that is padded and is 0
        # d[self.padded_mask]=float('nan')
        # return d

    def queryf(self, others: Sequence[Sequence[Var]], *queries: Sequence[Var]
               ) -> Callable[[Sequence[Factor]], Sequence[Tensor]]:
        equation = einsum.compile_obj_equation(cast(List[Sequence[Var]], [self.variables]) +
                                               list(others),
                                               queries, force_multi=True)

        def f(others: Sequence[Factor]) -> Sequence[Tensor]:
            # might be able to pull this out, but I want to make
            # sure that changes in e.g. usage are reflected
            dense = [self.dense]
            # any nans in any factor should be treated as a log(1)
            # meaning that it doesn't impact the product
            return einsum.log_einsum(equation, dense + [f.dense() for f in others])
        return f

    @ staticmethod
    def normalize(variables: Sequence[Var], tensor: Tensor) -> Tensor:
        num_dims = len(tensor.shape)
        num_batch_dims = len(tensor.shape) - len(variables)

        # normalize by subtracting out the sum of the last |V| dimensions
        variable_dims = list(range(num_batch_dims, num_dims))
        normalizer = torch.logsumexp(tensor, dim=variable_dims)
        tensor -= normalizer[(...,) + (None,) * (num_dims - num_batch_dims)]
        return tensor

    def free_energy(self, other_energy: Sequence[Factor], messages: Sequence[Factor]
                    ) -> Tensor:
        # TODO?: there is a way to do this with expectation semiring that would be general to
        # non-denseables
        variables = list(set(v
                             for f in [self, *other_energy]
                             for v in f.variables))
        log_belief = self.query([*other_energy, *messages], variables)[0]
        log_belief = DensableFactor.normalize(variables, log_belief)
        # positives = torch.logsumexp(log_belief.clamp_min(0) +
        #                             torch.where(log_belief >= 0,
        #                             log_belief.clamp_min(0).log(), 0.),
        #                             dim=variable_dims)
        # negatives = torch.logsumexp(-log_belief.clamp_max(0) +
        #                             torch.where(log_belief < 0,
        #                             (-log_belief.clamp_max(0)).log(), 0.),
        #                             dim=variable_dims)
        # entropy = torch.logsumexp(log_belief * log_belief.log(), dim=variable_dims)
        log_potentials, = self.query(other_energy, variables)
        belief = log_belief.exp()
        tensor = self.dense
        num_dims = len(tensor.shape)
        num_batch_dims = len(tensor.shape) - len(variables)
        # normalize by subtracting out the sum of the last |V| dimensions
        variable_dims = list(range(num_batch_dims, num_dims))
        entropy = torch.sum(belief * log_belief, dim=variable_dims)
        avg_energy = torch.sum(belief * log_potentials)
        return entropy - avg_energy
