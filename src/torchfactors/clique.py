
from abc import ABC, abstractmethod
from typing import Dict, Hashable, Sequence, Tuple

from torch.functional import Tensor

from .components.linear_factor import ShapedLinear
from .domain import Range
from .model import ParamNamespace
from .subject import Environment
from .variable import TensorVar, Var, VarUsage


# TODO: might be good to have sub environments like paramnamespaces
class CliqueModel(ABC):

    @abstractmethod
    def factors(self, env: Environment, params: ParamNamespace, *variables: Var, input: Tensor): ...


def make_binary_label_variables(env: Environment, *variables: Var, key: Hashable = None,
                                latent: bool = False
                                ) -> Dict[Tuple[Var, int], Var]:
    def binary_variable(v: Var, label: int):
        if latent:
            usage = v.usage.clone().where(v.usage == VarUsage.PADDING, VarUsage.LATENT)
        else:
            usage = v.usage.clone()
        out = TensorVar(
            domain=Range(2),
            usage=usage,
            tensor=(v.tensor == label).int())
        return out
    binary_variables: Dict[Tuple[Var, int], Var] = {
        (v, label): env.variable((v, label, key), lambda: binary_variable(v, label))
        for v in variables
        for label in range(len(v.domain))
    }
    return binary_variables


def make_binary_threshold_variables(env: Environment, *variables: Var, key: Hashable = None,
                                    latent: bool = False
                                    ) -> Dict[Tuple[Var], Var]:
    # TODO: allow different ways of thresholding
    def binary_variable(v: Var):
        if latent:
            usage = v.usage.clone().where(v.usage == VarUsage.PADDING, VarUsage.LATENT)
        else:
            usage = v.usage.clone()
        out = TensorVar(
            domain=Range(2),
            usage=usage,
            tensor=(v.tensor > len(v.domain) // 2).int())
        return out
    binary_variables: Dict[Tuple[Var], Var] = {
        v: env.variable((v, key), lambda: binary_variable(v))
        for v in variables
    }
    return binary_variables


def BinaryScoresModule(params: ParamNamespace, variables: Sequence[Var],
                       input: Tensor,
                       bias: bool = False):
    num_batch_dims = len(variables[0].shape) - 1

    def factory():
        out = ShapedLinear(input_shape=input.shape[num_batch_dims:],
                           output_shape=tuple(2 for v in variables),
                           bias=bias)
        return out
    out = params.module(factory)
    return out
