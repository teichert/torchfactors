
from typing import Optional

from torch.functional import Tensor

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

    def factors(self, env: Environment, params: ParamNamespace,
                *variables: Var, input: Optional[Tensor] = None,
                bias: bool = True):
        yield LinearFactor(params, *variables, input=input, bias=bias)
