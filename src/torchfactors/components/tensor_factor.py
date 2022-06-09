from typing import Callable, Optional

import torch

from ..factor import Factor
from ..types import ShapeType
from ..variable import Var


def float_zeros(shape: ShapeType):
    return torch.zeros(shape).float()


def linear_binary_to_ordinal_tensor(num_labels: int):
    out = float_zeros((2, num_labels))
    out[1, :] = torch.arange(num_labels) / (num_labels - 1)
    return out


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
        if len(tensor.shape) < len(self.shape) and tensor.shape == self.out_shape:
            tensor = tensor[(None,) * self.num_batch_dims].expand(self.shape)
        elif tensor.shape != self.shape:
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
