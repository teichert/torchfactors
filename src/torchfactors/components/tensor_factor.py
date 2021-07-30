from typing import Callable, Optional

import torch

from ..factor import Factor
from ..types import ShapeType
from ..variable import Var


def float_zeros(shape: ShapeType):
    return torch.zeros(shape).float()


class TensorFactor(Factor):
    r"""
    A factor that is fully specified by a single, fixed tensor. The tensor
    should assign a (log) score for each possible joint configuration of (the
    last dimension of ) variables. (reminder: all variables (batch dims not
    included) should match in all but the last dimension.) (For a paramterized
    tensor (e.g. bias), use a linear factor with no inputs.)

    """

    def __init__(self, *variables: Var,
                 tensor: Optional[torch.Tensor] = None,
                 init: Callable[[ShapeType], torch.Tensor] = float_zeros):
        super().__init__(variables)
        exemplar = self.variables[0].origin.tensor
        if tensor is None:
            tensor = init(self.shape)

        if tensor.shape != self.shape:
            try:
                tensor = tensor[None].expand(self.shape)
            except RuntimeError:
                raise ValueError("you didn't provide a tensor with the correct shape")
        self.tensor = tensor.to(exemplar.device)

    def dense_(self):
        return self.tensor


class Message(TensorFactor):
    r"""
    A TensorFactor that does not interact with variables nor do any caching.
    """

    def __init__(self, *variables: Var,
                 tensor: Optional[torch.Tensor] = None,
                 init: Callable[[ShapeType], torch.Tensor] = float_zeros):
        super().__init__(*variables, tensor=tensor, init=init)

    @property
    def dense(self):
        return self.tensor
