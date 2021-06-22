from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar

import torch
from torch import Tensor

from ..factor import DensableFactor
from ..model import ParamNamespace


@ dataclass
class LinearFactor(DensableFactor):
    __default: ClassVar[Tensor] = torch.tensor(0.0)
    params: ParamNamespace
    input: Tensor = __default
    bias: bool = True
    # input_dimensions: int = 1

    @ cached_property
    def in_shape(self):
        return tuple(self.input.shape[len(self.batch_shape):])

    @ cached_property
    def in_cells(self):
        return math.prod(self.in_shape)

    def dense_(self) -> Tensor:
        r"""returns a tensor that characterizes this factor;

        the factor's variable-domains dictate the number and
        size of the final dimensions.
        the variables themselves, then, know how many batch
        dimensions there are.
        """
        m = self.params.module(lambda:
                               torch.nn.Linear(
                                   in_features=self.in_cells,
                                   out_features=self.cells,
                                   bias=self.bias))
        input = self.input
        if not input.shape:
            input = input.expand((*self.batch_shape, 1))
        else:
            input = input.reshape((*self.batch_shape, -1))
        return m(input).reshape(self.shape)
