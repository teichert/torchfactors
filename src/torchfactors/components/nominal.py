
from typing import Optional

from torch import Tensor

from ..clique import CliqueModel
from ..model import ParamNamespace
from ..subject import Environment
from ..variable import Var
from .linear_factor import LinearFactor


class Nominal(CliqueModel):
    r"""
    A group of variables is modeled with a single linear factor: (optional TODO:
    allow minimal representation to avoid the one superfluous degree of freedom
    times number of features)
    """

    def __init__(self, bias: bool = True, minimal: bool = False, features: bool = True):
        super().__init__()
        self.has_bias = bias
        self.minimal = minimal
        self.features = features

    def factors(self, env: Environment, params: ParamNamespace,
                *variables: Var, input: Optional[Tensor] = None):
        yield LinearFactor(params, *variables,
                           input=input if getattr(self, 'features', True) else None,
                           bias=self.has_bias, minimal=getattr(self, 'minimal', False))
