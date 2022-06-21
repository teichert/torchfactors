from argparse import Namespace
import itertools
from tkinter.tix import Tree
from typing import Optional

import torch
from torch import Tensor
from torchfactors.components.at_least import KIsAtLeastJ
from torchfactors.components.linear_factor import LinearFactor, LinearTensor, MinimalLinear

from ..clique import (BinaryScoresModule, CliqueModel, ShapedLinear,
                      make_binary_label_variables)
from ..components.tensor_factor import TensorFactor
from ..factor import Factor
from ..model import ParamNamespace
from ..subject import Environment
from ..variable import Var
from torchfactors.components.binary import make_binary_factor, make_binary_tensor

from torchfactors import variable

class ProportionalOdds(CliqueModel):
    """
    Models a group of ordinal variables via a separate binary variable for
    each value of each variable (except the zero): the probability
    that the variable gets a label at least that large.
    """

    def factors(self, env: Environment, params: ParamNamespace,
                *variables: Var, input: Optional[Tensor] = None):

        # make a binary variable for each label of each variable (except 0)
        binary_variables = make_binary_label_variables(env, *variables)
        # weight_tensor = make_binary_tensor(params.namespace('binary-configs'), *variables, minimal=True, binary_bias=False, get_threshold=lambda _: 1)
        # weights are the same for all values (but different for different variable subsets)
        weight_tensor = LinearTensor(params.namespace('binary-config-weight'), *[binary_variables[v, 1] for v in variables], minimal=True, bias=False)(input)
        # bias is different for each configuration
        for configuration in itertools.product(*[range(1, len(v.domain)) for v in variables]):
            config_vars = [binary_variables[v, value] for v, value in zip(variables, configuration)]
            yield LinearFactor(params.namespace(f'config-bias:{configuration}'), *config_vars, input=None, bias=True, minimal=True)
            yield TensorFactor(*config_vars, tensor=weight_tensor)

        if len(variables) == 1:
            v = variables[0]
            for label in range(1, len(v.domain)):
                yield KIsAtLeastJ(v, binary_variables[v, label], label)

