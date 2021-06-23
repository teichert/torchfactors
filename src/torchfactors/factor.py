from __future__ import annotations

import math
from abc import abstractmethod
from functools import cached_property
from typing import Callable, Iterator, Sequence, Union

import torch
from torch import Tensor

from torchfactors import einsum
from torchfactors.types import ShapeType

from .utils import replace_negative_infinities
from .variable import Var


def check_queries(queries: Sequence[Sequence[Var]]):
    if not queries or isinstance(queries[0], Var):
        raise ValueError("each query needs to be a sequence of Vars "
                         "(did you forget the brackets around your single variable groups?) "
                         "consider using product_marginal() if you only have one variable "
                         "or if you want to just get the partition function.")


# @dataclass
class Factor:
    r"""
    A Factor has a domain which is defined by the set of variables it is
    concerned with. It is a function from any configuration of those variables
    to a mass value (i.e. we only consider non-negative values which is why we
    can operate in the log space).  To support inference among many factors, a
    factor need to know how to answer queries given other "denseable" factors
    as input and a set of (einsum style) queries to respond to.
    """

    def __init__(self, variables: Sequence[Var]):
        # if isinstance(variables, Var):
        #     variables = (variables,)
        for v in variables:
            if not isinstance(v, Var):
                raise TypeError(
                    f"it looks like you passed in a {type(v)} rather than a Var; "
                    "If the parameters comes after varargs, make sure to name it. "
                    "e.g. TensorFactor(v, tensor=t)")
        self.variables = variables
        for v in self.variables[1:]:
            if v.shape != self.variables[0].shape:
                raise ValueError("all variables must have the same shape (domains can vary)")

    def __iter__(self) -> Iterator[Var]:
        return iter(self.variables)

    def __len__(self) -> int:
        return len(self.variables)

    def product_marginal(self, query: Union[Sequence[Var], Var, None] = None,
                         other_factors: Sequence[Factor] = ()
                         ) -> Tensor:
        r"""
        convenience method for a single product_marginal query
        """
        if query is None:
            query = ()
        elif isinstance(query, Var):
            query = (query,)
        out, = self.product_marginals(query, other_factors=other_factors)
        return out

    def product_marginals(self, *queries: Sequence[Var], other_factors: Sequence[Factor] = ()
                          ) -> Sequence[Tensor]:
        r"""
        For each set of query variables, returns the marginal score for each
        configuration of those variables, with scores being computed as the product
        of this factor with the other `other_factors`.

        The partition function can be queried via an empty tuple of variables.
        If no queries are specified, then the partition function is queried.
        """
        check_queries(queries)
        return self.marginals_closure(*queries, other_factors=other_factors)()

    @staticmethod
    def out_shape_from_variables(variables: Sequence[Var]) -> ShapeType:
        return tuple([len(t.domain) for t in variables])

    @staticmethod
    def batch_shape_from_variables(variables: Sequence[Var]) -> ShapeType:
        first = variables[0]
        return tuple(first.tensor.shape)

    @staticmethod
    def shape_from_variables(variables: Sequence[Var]) -> ShapeType:
        return tuple([*Factor.batch_shape_from_variables(variables),
                      *Factor.out_shape_from_variables(variables)])

    @cached_property
    def out_shape(self):
        r"""
        The shape of the output configuration scores (the joint sizes from the
        last dimension of each variable input to the factor).
        """
        return Factor.out_shape_from_variables(self.variables)

    @cached_property
    def shape(self):
        r"""
        Returns the shape of the (possibly implicit) tensor that would represent this tensor
        """
        return tuple([*self.batch_shape, *self.out_shape])

    @cached_property
    def cells(self):
        r"""
        The number of cells in the (possibly implicit) tensor that would
        represent this tensor
        """
        return math.prod(self.shape)

    @cached_property
    def out_cells(self):
        r"""
        The number of cells in the (possibly implicit) output for each batch
        element
        """
        return math.prod(self.out_shape)

    @cached_property
    def batch_cells(self):
        r"""
        The number of elements in a single batch (in contrast to the number of
        variable configurations per element).
        """
        return math.prod(self.batch_shape)

    @cached_property
    def batch_shape(self):
        r"""
        The shape of the dimensions of the implicit tensor corresponding to
        various elements in a single batch as opposed to alternative
        configurations of a single element of the batch. (All dimensions of the
        variables are batch dimensions with the actual configuration being
        identified by the value [an additional dimension].  The batch dimensions
        should be consistent across variables, but the domain of the variables
        [and thus the output dimensions] need not be consitent.)
        """
        # should be the same for all variables (maybe worth checking?)
        return Factor.batch_shape_from_variables(self.variables)

    @property
    def num_batch_dims(self):
        r"""
        The number of dimensions dedicated to distinguishing elements of a
        single batch (vs distinguishing configurations of a single element).
        """
        return len(self.batch_shape)

    @abstractmethod
    def dense_(self) -> Tensor: ...

    @property
    def dense(self) -> Tensor:
        r"""
        A tensor representing the factor (should have the same shape as
        the factor and dimensions should correspond to the variables in the same
        order as given by the factor).
        """
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

    def marginals_closure(self, *queries: Sequence[Var], other_factors: Sequence[Factor] = ()
                          ) -> Callable[[], Sequence[Tensor]]:
        r"""
        Given a set of other factors and a set of product_marginal queries,
        returns a function that recomputes the corresponding marginals for each
        query given the current values of the factors.

        """
        check_queries(queries)
        batch_dims = [object() for _ in range(self.num_batch_dims)]

        def with_batch_dims(objs: Sequence[object]) -> Sequence[object]:
            return tuple([*batch_dims, *objs])

        equation = einsum.compile_obj_equation(
            [with_batch_dims(self.variables)] +
            [with_batch_dims(other.variables)
             for other in other_factors],
            [with_batch_dims(q) for q in queries], force_multi=True)

        def f() -> Sequence[Tensor]:
            # might be able to pull this out, but I want to make
            # sure that changes in e.g. usage are reflected
            # any nans in any factor should be treated as a log(1)
            # meaning that it doesn't impact the product
            input_tensors = [self.dense] + [f.dense
                                            for f in other_factors]
            return einsum.log_einsum(equation, *input_tensors)
        return f

    @ staticmethod
    def normalize(tensor: Tensor, num_batch_dims=0) -> Tensor:
        r"""
        for each of the batch entries, normalizes the following dimensions so that the
        values logsumexp up to 0.
        """
        num_dims = len(tensor.shape)

        # normalize by subtracting out the sum of the last |V| dimensions
        variable_dims = list(range(num_batch_dims, num_dims))
        normalizer = torch.logsumexp(tensor, dim=variable_dims)
        tensor -= normalizer[(...,) + (None,) * (num_dims - num_batch_dims)]
        return tensor

    def free_energy(self,
                    other_energy: Sequence[Factor] = (),
                    messages: Sequence[Factor] = ()
                    ) -> Tensor:
        r"""
        Returns an estimate of the additive contribution of this factor to the
        total free energy of a collection of factors.

        The exact free energy of the system is the negative log partition
        function.  The partition function is the constant factor Z that would
        normalize the scores to sum to one.  With appropriate grouping of
        factors, it is possible to compute the exact free energy of the system
        in terms of free energies of groups by subtracting out double-counted
        groups.  The free energy of a particular group with respect to the
        entire system is a function of the scores of each joint configuration of
        the group as well as an estimate of the marginal probability of each
        group configuration under the full system.

        In this function, the group configuration scores are specified via
        "commanding factor" (self) and the `other_energy` factors whier are all
        multiplied together to produce a score for each joint configuration. The
        marginal probabilities are additionally specified by the `messages`
        which are multiplied to the commander and other_energy factors, with the
        output being normalized to produce a distribution over the
        configurations.

        Given these computed configuration scores and marginal probabilities
        (known as beliefs).  The group free energy is simply the entropy of the
        belief minus the average log score.  (Be aware that our ipmlementation
        deals with scores and in log-space to begin with.) A separate
        free_energy is computed for each batch_dimension so the shape of the
        output tensor should be the batch_shape.

        This computation is straight forward for dense factors, but can require
        care for sparse (a.k.a. structured) factors (e.g.
        projective-dependency-tree factor, or CFG-Tree factor). Thus,
        implementing such a factor requires implementing a mechanism for
        computing this quantity.
        """
        # TODO?: there is a way to do this with expectation semiring that would be general to
        # non-denseables
        variables = list(set(v
                             for f in [self, *other_energy]
                             for v in f.variables))
        log_belief = self.product_marginal(variables, other_factors=[*other_energy, *messages])
        log_belief = Factor.normalize(log_belief, num_batch_dims=self.num_batch_dims)
        # positives = torch.logsumexp(log_belief.clamp_min(0) +
        #                             torch.where(log_belief >= 0,
        #                             log_belief.clamp_min(0).log(), 0.),
        #                             dim=variable_dims)
        # negatives = torch.logsumexp(-log_belief.clamp_max(0) +
        #                             torch.where(log_belief < 0,
        #                             (-log_belief.clamp_max(0)).log(), 0.),
        #                             dim=variable_dims)
        # entropy = torch.logsumexp(log_belief * log_belief.log(), dim=variable_dims)
        log_potentials = self.product_marginal(variables, other_factors=other_energy)
        belief = log_belief.exp()
        # tensor = self.dense
        # num_dims = len(tensor.shape)
        # num_batch_dims = len(tensor.shape) - len(variables)
        # # normalize by subtracting out the sum of the last |V| dimensions
        # variable_dims = list(range(num_batch_dims, num_dims))
        variable_dims = list(range(self.num_batch_dims, len(self.shape)))
        entropy = torch.sum(belief * replace_negative_infinities(log_belief), dim=variable_dims)
        avg_energy = torch.sum(
            belief * replace_negative_infinities(log_potentials), dim=variable_dims)
        return entropy - avg_energy


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
