
from torch.functional import Tensor

from ..clique import CliqueModel
from ..model import ParamNamespace
from ..subject import Environment
from ..variable import Var


class LatentBinary(CliqueModel):

    def factors(self, env: Environment, params: ParamNamespace,
                *variables: Var, input: Tensor):
        # make a binary variable for each binary configuration
        pass
