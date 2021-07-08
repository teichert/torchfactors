
from torch.functional import Tensor

from ..clique import CliqueModel
from ..model import ParamNamespace
from ..subject import Environment
from ..variable import Var
from .linear_factor import LinearFactor


class Nominal(CliqueModel):

    def factors(self, x: Environment, params: ParamNamespace,
                *variables: Var, input: Tensor):
        yield LinearFactor(params, *variables, input=input)
