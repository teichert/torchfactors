from typing import Callable, Optional, Sequence, Union

import torch
from torch import Tensor
from torchfactors.types import ShapeType
from torchfactors.variable import Var

from ..factor import Factor


class TensorFactor(Factor):
    r"""
    A factor that is fully specified by a single, fixed tensor. The tensor
    should assign a (log) score for each possible joint configuration of (the
    last dimension of ) variables. (reminder: all variables should match in all
    but the last dimension.) (For a paramterized tensor (e.g. bias), use a
    linear factor with no inputs.)

    """

    def __init__(self, variables: Union[Var, Sequence[Var]],
                 tensor: Optional[Tensor] = None,
                 init: Callable[[ShapeType], Tensor] = torch.zeros):
        super().__init__(variables)
        if tensor is None:
            tensor = init(self.shape)
        if tensor.shape != self.shape:
            raise ValueError("you didn't provide a tensor with the correct shape")
        self.tensor = tensor

    def dense_(self):
        return self.tensor
