from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import Tensor

from ..factor import DensableFactor

# Note: linear factor with no inputs is the version of this that gets the
# parameters from the model


@ dataclass
class TensorFactor(DensableFactor):
    AUTO: ClassVar[Tensor] = torch.tensor(0.0)
    tensor: Tensor = AUTO

    def dense_(self):
        return self.tensor

    def __post_init__(self):
        if self.tensor is TensorFactor.AUTO:
            self.tensor = torch.zeros(
                *[len(v.domain) for v in self.variables])
