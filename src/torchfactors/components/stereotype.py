
import torch
import torchfactors as tx
from torch.functional import Tensor

from ..model import ParamNamespace
from ..subject import Environment
from ..variable import Var
from .linear_factor import BinaryScoresModule, LinearFactor


class Stereotype(tx.CliqueModel):

    def factors(self, x: Environment, params: ParamNamespace,
                *variables: Var, input: Tensor, linear: bool = True):

        # no input; just bias
        yield LinearFactor(params.namespace('group-bias'), *variables)

        binary_scores_module = BinaryScoresModule(params.namespace('group-score'), variables)
        binary_scores = binary_scores_module(input)

        if linear:
            one_scale = [torch.linspace(0, 1, len(v.domain)) for v in variables]
            scales = tx.utils.outer(one_scale)
        else:
            scales = params.namespace('group-scale').parameter(tuple(
                len(v.domain) for v in variables))
        scores = tx.utils.stereotype(scales, binary_scores)
        stereotype = tx.TensorFactor(*variables, tensor=scores)
        yield stereotype
