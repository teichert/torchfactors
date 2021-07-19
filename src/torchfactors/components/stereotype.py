
from typing import Optional

import torch
import torchfactors as tx
from torch.functional import Tensor

from ..clique import BinaryScoresModule, CliqueModel
from ..model import ParamNamespace
from ..subject import Environment
from ..variable import Var
from .linear_factor import LinearFactor


class Stereotype(CliqueModel):

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
                scales = tx.utils.outer(*one_scale)
            else:
                scales = params.namespace('group-scale').parameter(tuple(
                    len(v.domain) for v in variables))
            scores = tx.utils.stereotype(scales, binary_scores)
            stereotype = tx.TensorFactor(*variables, tensor=scores)
            yield stereotype
