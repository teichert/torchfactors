
from typing import Optional

from torch import Tensor
from torchfactors import TensorFactor
from torchfactors.components.tensor_factor import \
    linear_binary_to_ordinal_tensor

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

    def __init__(self, latent: bool = True, linear: bool = False):
        super().__init__()
        self.latent = latent
        self.linear = linear

    # TODO: allow no bias?
    def factors(self, env: Environment, params: ParamNamespace,
                *variables: Var, input: Optional[Tensor] = None):
        binary_variables = make_binary_threshold_variables(env, *variables, latent=self.latent)
        yield LinearFactor(params.namespace('binary-group'),
                           *binary_variables.values(), input=input)
        for i, (ordinal, binary) in enumerate(binary_variables.items()):
            if self.linear:
                t = linear_binary_to_ordinal_tensor(len(ordinal.domain))
                def factor(): return TensorFactor(binary, ordinal, tensor=t)
            else:
                def factor(): return LinearFactor(params.namespace((i, 'binary-to-ordinal')), binary, ordinal, bias=True)
            yield env.factor((ordinal, 'binary-to-ordinal'), factor)
