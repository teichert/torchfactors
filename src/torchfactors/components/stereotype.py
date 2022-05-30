
from typing import Optional

import torch
import torchfactors as tx
from torch import Tensor

from ..clique import BinaryScoresModule, CliqueModel
from ..model import ParamNamespace
from ..subject import Environment
from ..variable import Var
from .linear_factor import LinearFactor


class Stereotype(CliqueModel):
    r"""
    Models an ordinal variable with a log-linear score that is then scaled and
    then offset by a label-specific coefficient and bias.  Likewise for
    higher-order configurations, each full configuration aligns with some scale
    to each binary configuration and these are combined and then transformed by
    a configuration-specific coefficient and bias.

    TODO: partially ordered: something similar could be applied even if some
    variables or portions of variables require more than a single score.
    Essentially, each dimension (i.e. subset of variable-label pairs) that needs
    a separate scalar, would be treated as a separate output binary variable. (I
    need to work this out more)

    (This is more motivation for having some concept of "type", "name", or
    "properties" to come in as part of the variable.)
    """

    def __init__(self, linear: bool = True):
        self.linear = linear

    def factors(self, x: Environment, params: ParamNamespace,
                *variables: Var, input: Optional[Tensor] = None):

        # no input; just bias
        yield LinearFactor(params.namespace('group-bias'), *variables)
        if input is not None:
            binary_scores_module = BinaryScoresModule(params.namespace('group-score'), variables,
                                                      input=input, bias=False)
            binary_scores = binary_scores_module(input)

            if self.linear:
                one_scale = [torch.linspace(0, 1, len(v.domain)) for v in variables]
                scales = tx.utils.outer(one_scale)
            else:
                scales = params.namespace('group-scale').parameter(tuple(
                    len(v.domain) for v in variables))
            scores = tx.utils.stereotype(scales, binary_scores)
            stereotype = tx.TensorFactor(*variables, tensor=scores)
            yield stereotype
