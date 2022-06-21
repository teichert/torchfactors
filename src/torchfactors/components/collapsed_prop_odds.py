from argparse import Namespace
import itertools
from tkinter.tix import Tree
from typing import Optional

import torch
from torch import Tensor
from torchfactors.components.at_least import KIsAtLeastJ
from torchfactors.components.linear_factor import LinearFactor, MinimalLinear

from ..clique import (BinaryScoresModule, CliqueModel, ShapedLinear,
                      make_binary_label_variables)
from .tensor_factor import TensorFactor
from ..factor import Factor
from ..model import ParamNamespace
from ..subject import Environment
from ..variable import Var
from torchfactors.components.binary import make_binary_factor

class CollapsedProporionalOdds(CliqueModel):
    """
    Models a group of ordinal variables via a separate binary variable for
    each value of each variable (except the zero): the probability
    that the variable gets a label at least that large.
    """

    def factors(self, env: Environment, params: ParamNamespace,
                *variables: Var, input: Optional[Tensor] = None):
        # each subset of variables gets a separate weight based on the input
        yield make_binary_factor(params.namespace('binary-configs'), *variables, input=input, minimal=True, binary_bias=False, get_threshold=lambda _: 1)
        # and each configuration gets a separate bias
        yield LinearFactor(params.namespace('ordinal-bias'), *variables, input=None, bias=True, minimal=True)
