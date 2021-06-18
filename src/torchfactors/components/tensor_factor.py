from dataclasses import InitVar, dataclass
from typing import Optional

import torch
from torch import Tensor

from ..factor import DensableFactor


@ dataclass
class TensorFactor(DensableFactor):
    __tensor: InitVar[Optional[Tensor]] = None

    def dense(self):
        return self.tensor

    def __post_init__(self, tensor: Optional[Tensor]):
        if tensor is not None:
            self.__tensor = tensor
        else:
            self.__tensor = torch.zeros(
                *[len(v.domain) for v in self.variables])

    @ property
    def tensor(self):
        # in here, we need to only allow the values consistent
        # with clamped or observed variables (competitors go to log(0));
        # and then we need to take any padded inputs
        # as log(1)
        #
        return self.__tensor
