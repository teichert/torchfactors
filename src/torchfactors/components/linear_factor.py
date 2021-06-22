from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import Sequence, Union

import torch
from torch import Tensor

from torchfactors.factor import Factor
from torchfactors.model import ParamNamespace
from torchfactors.variable import Var


@ dataclass
class LinearFactor(Factor):

    def __init__(self,
                 variables: Union[Var, Sequence[Var]],
                 params: ParamNamespace,
                 input: Tensor = torch.tensor(0.0),
                 bias: bool = True):
        super().__init__(variables)
        self.input = input
        self.bias = bias
        self.params = params

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
