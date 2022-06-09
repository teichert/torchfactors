from __future__ import annotations

import math
from abc import abstractmethod
from typing import Iterator, List, Sequence, Tuple, Union

import torch
from torch import Tensor

from .einsum import log_dot, slow_log_dot
from .types import ShapeType
from .utils import logsumexp, outer_or
from .variable import Var, VarUsage, possibility


def check_queries(queries: Sequence[Union[Var, Sequence[Var]]]):
    if not queries or isinstance(queries[0], Var):
        raise ValueError("each query needs to be a sequence of Vars "
                         "(did you forget the brackets around your single variable groups?) "
                         "consider using product_marginal() if you only have one variable "
                         "or if you want to just get the partition function.")


@torch.jit.script
def adjust(tensor: Tensor, var_infos: List[Tuple[Tensor, Tensor, List[int]]],
           ANNOTATED: int,
           LATENT: int,
           PADDING: int,
           num_batch_dims: int):  # pragma: no cover
    padding = outer_or([
        (usage == PADDING).unsqueeze(-1).expand(shape)
        for usage, _, shape in var_infos], num_batch_dims)
    not_possible = outer_or([
        possibility(usage, values, shape, LATENT, ANNOTATED).logical_not()
        for usage, values, shape in var_infos], num_batch_dims)
    return tensor.masked_fill(
        padding,
        0.0
    ).masked_fill(
        not_possible,
        float('-inf')
    )


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
        for v in variables:
            if not isinstance(v, Var):
                raise TypeError(
                    f"it looks like you passed in a {type(v)} rather than a Var; "
                    "If the parameters comes after varargs, make sure to name it. "
                    "e.g. TensorFactor(v, tensor=t)")
        if len(set(variables)) < len(variables):
            raise ValueError("It looks like you've used the same variable more than once in "
                             "this factor (sorry, this is not supported yet). Did you forget "
                             "ellipses when you indexed into a variable (e.g. v[:-1], v[1:] "
                             "instead of v[..., :-1], v[..., 1:])?")
        self.variables = variables
        for v in self.variables[1:]:
            if v.shape[:self.num_batch_dims] != self.variables[0].shape[:self.num_batch_dims]:
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

    @property
    def out_shape(self):
        r"""
        The shape of the output configuration scores (the joint sizes from the
        last dimension of each variable input to the factor).
        """
        return Factor.out_shape_from_variables(self.variables)

    @property
    def shape(self):
        r"""
        Returns the shape of the (possibly implicit) tensor that would represent this tensor
        """
        return tuple([*self.batch_shape, *self.out_shape])

    @property
    def cells(self):
        r"""
        The number of cells in the (possibly implicit) tensor that would
        represent this tensor
        """
        return math.prod(self.shape)

    @property
    def batch_cells(self):
        r"""
        The number of elements in a single batch (in contrast to the number of
        variable configurations per element).
        """
        return math.prod(self.batch_shape)

    @property
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
    def dense_(self) -> Tensor: ...  # pragma: no cover

    def prime(self):
        """ensures that all parameters are loaded for this factor"""
        self.dense_()

    @property
    def dense(self) -> Tensor:
        r"""
        A tensor representing the factor (should have the same shape as
        the factor and dimensions should correspond to the variables in the same
        order as given by the factor).
        TODO: ISSUES: it might be cleaner and more economical to handle clamping
        by introducing additional factors, but in some ways it is nice to have the
        number of factors not change.  Another issue is that non-densable factors will
        need to separately implement this logic :(
        """
        d = self.dense_()
        if d.shape != self.shape:
            d = d[None].expand(self.shape)
        var_infos = [(v.usage_readonly, v.tensor, v.marginal_shape) for v in self.variables]
        return adjust(d, var_infos,
                      VarUsage.ANNOTATED,
                      VarUsage.LATENT,
                      VarUsage.PADDING,
                      self.num_batch_dims)

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
        batch_ids = [str(d) for d in range(self.num_batch_dims)]

        def with_batch_ids(vs: Sequence[Var]):
            out = [*batch_ids, *[str(v.hash_key()) for v in vs]]
            return out

        input_tensors = [(self.dense, with_batch_ids(self.variables)), *[
            (f.dense, with_batch_ids(f.variables)) for f in other_factors]]
        labeled_queries = [with_batch_ids(q) for q in queries]

        # try:
        out = log_dot(input_tensors, labeled_queries, nan_to_num=True)
        # except:
        # out = slow_log_dot(input_tensors, labeled_queries, nan_to_num=True)
        return out

    @ staticmethod
    def normalize(tensor: Tensor, num_batch_dims=0) -> Tensor:
        r"""
        for each of the batch entries, normalizes the following dimensions so that the
        values logsumexp up to 0.
        """
        num_dims = len(tensor.shape)

        # normalize by subtracting out the sum of the last |V| dimensions
        variable_dims = list(range(num_batch_dims, num_dims))
        normalizer = logsumexp(tensor, dim=variable_dims)
        return tensor - normalizer[(...,) + (None,) * (num_dims - num_batch_dims)]

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
        log_potentials = self.product_marginal(variables, other_factors=other_energy)
        belief = log_belief.exp()
        variable_dims = list(range(self.num_batch_dims, len(self.shape)))
        masked_belief = belief.masked_fill(belief <= 0, 1.0)
        masked_log_belief = log_belief.masked_fill(belief <= 0, 0.0)
        masked_log_potentials = log_potentials.masked_fill(belief <= 0, 0.0)
        return (masked_belief * (masked_log_belief - masked_log_potentials)).sum(
            dim=variable_dims)
