
from gc import get_threshold
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
from .linear_factor import LinearFactor, LinearTensorAux, MinimalLinear, ShapedLinear


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
    def factors_(self, env: Environment, params: ParamNamespace,
                 *variables: Var, input: Optional[Tensor] = None):
        if self.latent:
            # TODO: no key is being passed here so there there is coupling
            # between  (could be a source of bugs)
            ordinal_to_binary_variable = make_binary_threshold_variables(
                env, *variables, latent=True)
            yield LinearFactor(params.namespace('latent-binary'),
                               *ordinal_to_binary_variable.values(),
                               input=input, minimal=self.minimal)
            for i, (ordinal, binary) in enumerate(ordinal_to_binary_variable.items()):
                if self.linear:
                    t = linear_binary_to_ordinal_tensor(len(ordinal.domain))
                    def factor(): return TensorFactor(binary, ordinal, tensor=t)
                else:
                    def factor(): return LinearFactor(params.namespace((i, 'binary-to-ordinal')),
                                                      binary, ordinal, bias=True,
                                                      minimal=self.minimal)
                yield env.factor((ordinal, 'binary-to-ordinal'), factor)
        else:
            yield make_binary_factor(params.namespace('binary'),
                minimal=self.minimal, *variables, input=input,
                binary_bias=True, get_threshold=lambda v: binarization(len(v.domain))[0])


def make_binary_factor(params, *variables, **kwargs):
    return TensorFactor(*variables, tensor=make_binary_tensor(params, *variables, **kwargs))


def make_binary_tensor(params, *variables, input, minimal: bool, binary_bias: bool, get_threshold):
    binary_tensor = LinearTensorAux(params, *variables, out_shape=(2,) * len(variables),
        bias=binary_bias, minimal=minimal)(input)
    batch_dims = len(binary_tensor.shape) - len(variables)
    for i, v in enumerate(variables):
        dim = batch_dims + i
        num_negative = get_threshold(v)
        num_positive = len(v.domain) - num_negative
        repeats = torch.tensor([num_negative, num_positive])
        binary_tensor = binary_tensor.repeat_interleave(repeats, dim=dim)
    return binary_tensor

def binarization(domain_size: int):
    """
    Returns the number of the labels to be considered as negative and the number
    to be considered as positive.
    """
    num_positive = domain_size // 2
    num_negative = domain_size - num_positive
    return num_negative, num_positive
