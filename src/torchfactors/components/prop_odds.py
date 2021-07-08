import itertools
from typing import Dict, Tuple

import torch
from torch.functional import Tensor
from torchfactors.components.tensor_factor import TensorFactor
from torchfactors.factor import Factor

from ..clique import CliqueModel
from ..domain import Range
from ..model import ParamNamespace
from ..subject import Environment
from ..variable import TensorVar, Var
from .linear_factor import BinaryScoresModule, ShapedLinear

# class LatentBinary(tx.CliqueModel):

#     def factors(self, x: Environment, params: ParamNamespace,
#                 *variables: Var, input: Tensor):

#         # make a binary variable for each binary configuration
#         pass


class ProportionalOdds(CliqueModel):

    def factors(self, env: Environment, params: ParamNamespace,
                *variables: Var, input: Tensor):

        # make a binary variable for each label of each variable
        def binary_variable(v: Var, label: int):
            out = TensorVar(
                domain=Range(2),
                usage=v.usage.clone(),
                tensor=(v.tensor == label).int())
            return out
        binary_variables: Dict[Tuple[Var, int], Var] = {
            (v, label): env.variable((v, label), lambda: binary_variable(v, label))
            for v in variables
            for label in range(len(v.domain))
        }
        binary_scores_module = BinaryScoresModule(
            params.namespace('binary-scores'), variables, input=input)
        binary_scores = binary_scores_module(input)

        def bias_module_factory():
            return ShapedLinear(output_shape=Factor.out_shape_from_variables(variables), bias=True)
        bias_module = params.namespace('full-bias').module(bias_module_factory)
        bias = bias_module(None)
        for config in itertools.product(*[range(1, len(v.domain)) for v in variables]):
            config_variables = [
                binary_variables[v, label] for v, label in zip(variables, config)
            ]
            # the config-specific bias is applied to whether or not all of the
            # corresponding binary variables is true
            config_bias = torch.sparse_coo_tensor(
                torch.ones((len(variables), 1)),
                [bias[config]], binary_scores.shape)
            yield TensorFactor(*config_variables, tensor=binary_scores + config_bias)
