from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import Tensor

from ..factor import DensableFactor


@ dataclass
class TensorFactor(DensableFactor):
    r"""
    A factor that is fully specified by a single, fixed tensor. The tensor
    should assign a (log) score for each possible joint configuration of (the
    last dimension of ) variables. (reminder: all variables should match in all
    but the last dimension.) (For a paramterized tensor (e.g. bias), use a
    linear factor with no inputs.)

    """

    AUTO: ClassVar[Tensor] = torch.tensor(0.0)
    tensor: Tensor = AUTO

    def dense_(self):
        return self.tensor

    def __post_init__(self):
        if self.tensor is TensorFactor.AUTO:
            self.tensor = torch.zeros(
                *[len(v.domain) for v in self.variables])
