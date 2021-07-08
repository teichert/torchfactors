
from torch.functional import Tensor

from ..clique import CliqueModel, make_binary_variables
from ..model import ParamNamespace
from ..subject import Environment
from ..variable import Var


class LatentBinary(CliqueModel):

    def factors(self, env: Environment, params: ParamNamespace,
                *variables: Var, input: Tensor):
        # binary variables are latent; mapping to observed ordinals is learned
        binary_variables = make_binary_variables(env, *variables, latent=True)
