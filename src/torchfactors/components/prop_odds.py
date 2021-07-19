import itertools

import torch
from torch.functional import Tensor
from torchfactors.components.at_least import KIsAtLeastJ

from ..clique import (BinaryScoresModule, CliqueModel, ShapedLinear,
                      make_binary_label_variables)
from ..components.tensor_factor import TensorFactor
from ..factor import Factor
from ..model import ParamNamespace
from ..subject import Environment
from ..variable import Var


class ProportionalOdds(CliqueModel):
    """
    Models a group of ordinal variables via a separate binary variable for
    each value of each variable (except the zero): the probability
    that the variable gets a label at least that large.
    """

    def factors(self, env: Environment, params: ParamNamespace,
                *variables: Var, input: Tensor):

        # make a binary variable for each label of each variable
        binary_variables = make_binary_label_variables(env, *variables)
        # produce a score using the given input
        binary_scores_module = BinaryScoresModule(
            params.namespace('binary-scores'), variables, input=input)
        binary_scores = binary_scores_module(input)

        # we likewise, want to create a separate bias for each full configuration
        def bias_module_factory():
            return ShapedLinear(output_shape=Factor.out_shape_from_variables(variables), bias=True)
        bias_module = params.namespace('full-bias').module(bias_module_factory)

        # since it is only a bias, it doesn't matter what we pass in
        bias = bias_module(torch.tensor(0.0))

        # the bias for a particular config is actually applied in favor
        # of having all of the given set of binary variables on; skip 0
        # for identifiability
        for config in itertools.product(*[range(1, len(v.domain)) for v in variables]):
            # the binary variables corresponding to this ordinal config
            config_variables = [
                binary_variables[v, label] for v, label in zip(variables, config)
            ]
            # a tensor of the right shape with the given bias
            config_bias = torch.sparse_coo_tensor(
                torch.ones((len(variables), 1)),
                [bias[config]], binary_scores.shape).log()
            yield TensorFactor(*config_variables, tensor=binary_scores + config_bias)

        # deterministically set the ordinals given the values of the binaries
        for v in variables:
            # deterministically say that if the binary 'j' is off, then
            # the values k >= j are all impossible
            for label in range(1, len(v.domain)):
                yield KIsAtLeastJ(v, binary_variables[v, label], label)
            # yield VIsKAndAtLeastB(v, [
            #     binary_variables[v, label] for label in range(1, len(v.domain))
            # ])
