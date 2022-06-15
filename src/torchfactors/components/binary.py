
from typing import Optional

import torch
from torch import Tensor
from torchfactors import TensorFactor
from torchfactors.components.tensor_factor import \
    linear_binary_to_ordinal_tensor

from ..clique import CliqueModel, make_binary_threshold_variables
from ..model import ParamNamespace
from ..subject import Environment
from ..variable import Var
from .linear_factor import LinearFactor, MinimalLinear, ShapedLinear


class Binary(CliqueModel):
    """
    uses observed auxiliary binary variables thresholded at half of the labels;
    the mapping from binaries to observed actual lables is also learned
    """

    def __init__(self, latent: bool = True, linear: bool = False, minimal=True):
        super().__init__()
        self.latent = latent
        self.linear = linear
        self.minimal = minimal

    # TODO: allow no bias?
    def factors(self, env: Environment, params: ParamNamespace,
                *variables: Var, input: Optional[Tensor] = None):
        # yield
        # print(binary_tensor)
        if self.latent:
            binary_variables = make_binary_threshold_variables(env, *variables, latent=True)
            yield LinearFactor(params.namespace('latent-binary'), *binary_variables, input=input,
                               minimal=self.minimal)
            # yield TensorFactor(*binary_variables.values(), tensor=binary_tensor)
            for i, (ordinal, binary) in enumerate(binary_variables.items()):
                if self.linear:
                    t = linear_binary_to_ordinal_tensor(len(ordinal.domain))
                    def factor(): return TensorFactor(binary, ordinal, tensor=t)
                else:
                    def factor(): return LinearFactor(params.namespace((i, 'binary-to-ordinal')),
                                                      binary, ordinal, bias=True, minimal=self.minimal)
                yield env.factor((ordinal, 'binary-to-ordinal'), factor)
        else:
            make_binary_tensor = params.namespace('binary').module(
                (MinimalLinear if self.minimal else ShapedLinear),
                output_shape=(2,) * len(variables))
            binary_tensor = make_binary_tensor(input)
            for dim, v in enumerate(variables):
                num_positive = len(v.domain) // 2
                num_negative = len(v.domain) - num_positive
                repeats = torch.tensor([num_negative, num_positive])
                binary_tensor = binary_tensor.repeat_interleave(repeats, dim=dim)
            yield TensorFactor(*variables, tensor=binary_tensor)
