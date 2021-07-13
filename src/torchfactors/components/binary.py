
from torch.functional import Tensor

from ..clique import CliqueModel, make_binary_threshold_variables
from ..model import ParamNamespace
from ..subject import Environment
from ..variable import Var
from .linear_factor import LinearFactor


class Binary(CliqueModel):
    """
    uses observed auxiliary binary variables thresholded at half of the labels;
    the mapping from binaries to observed actual lables is also learned
    """

    def __init__(self, latent: bool = True):
        self.latent = latent

    def factors(self, env: Environment, params: ParamNamespace,
                *variables: Var, input: Tensor):
        binary_variables = make_binary_threshold_variables(env, *variables, latent=self.latent)
        yield LinearFactor(params.namespace('binary-group'),
                           *binary_variables.values(), input=input)
        for ordinal, binary in binary_variables.items():
            yield env.factor(ordinal, lambda: LinearFactor(params.namespace('binary-to-ordinal'),
                                                           binary, ordinal, input=None))
