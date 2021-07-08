from __future__ import annotations

import math
from functools import cached_property
from typing import Optional, Union, cast

import torch
from torch.functional import Tensor

from ..factor import Factor
from ..model import ParamNamespace
from ..types import ShapeType
from ..variable import Var


class OptionalBiasLinear(torch.nn.Module):
    r"""
    Allows the output to ignore the input (bias-only),
    but otherwise, works like a regular linear
    """

    def __init__(self, input_features: int, output_features: int,
                 bias: bool = True):
        super().__init__()
        self.input_features = input_features
        self.bias_only: Union[torch.nn.parameter.Parameter, torch.Tensor, None] = None
        self.with_features: Optional[torch.nn.Module] = None
        if input_features == 0:
            if bias:
                self.bias_only = torch.nn.parameter.Parameter(
                    torch.rand(output_features)
                )
            else:
                self.bias_only = torch.tensor(0.0).expand((output_features,))
        else:
            self.with_features = torch.nn.Linear(
                input_features, output_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        if self.bias_only is not None:
            # out = self.bias_only.expand(*x.shape[:-1], len(self.bias_only))
            out = self.bias_only
        else:
            out = cast(torch.nn.Module, self.with_features)(x)
        return out


class ShapedLinear(torch.nn.Module):

    def __init__(self, output_shape: ShapeType,
                 bias: bool = True, input_shape: ShapeType = None):
        super().__init__()
        self.input_shape = input_shape
        if input_shape is None:
            input_features = 0
        else:
            input_features = math.prod(input_shape)
        self.output_shape = output_shape
        output_features = math.prod(output_shape)
        self.wrapped_linear = OptionalBiasLinear(input_features, output_features,
                                                 bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        flattened_in: Tensor
        if self.input_shape is None:
            flattened_in = x
        else:
            flattened_in = x.flatten(len(x.shape) - len(self.input_shape))
        out: Tensor = self.wrapped_linear(flattened_in)
        reshaped = out.unflatten(-1, self.output_shape)
        return reshaped


class LinearFactor(Factor):

    def __init__(self,
                 params: ParamNamespace,
                 *variables: Var,
                 input: torch.Tensor = None,
                 bias: bool = True):
        super().__init__(variables)
        self.input = input
        self.bias = bias
        self.params = params
        if input is not None and input.shape:
            if (len(input.shape) < self.num_batch_dims or
                    input.shape[:self.num_batch_dims] != self.batch_shape):
                raise ValueError("prefix dimensions of input must match batch_dims")

    @cached_property
    def in_shape(self) -> Optional[ShapeType]:
        if self.input is None:
            return None
        else:
            return self.input.shape[len(self.batch_shape):]

    def dense_(self) -> torch.Tensor:
        r"""returns a tensor that characterizes this factor;

        the factor's variable-domains dictate the number and
        size of the final dimensions.
        the variables themselves, then, know how many batch
        dimensions there are.
        """
        m = self.params.module(lambda:
                               ShapedLinear(
                                   output_shape=self.out_shape,
                                   input_shape=self.in_shape,
                                   bias=self.bias))
        # m = self.params.module(lambda:
        #                        torch.nn.Linear(
        #                            in_features=self.in_cells,
        #                            out_features=self.out_cells,
        #                            bias=self.bias))
        # input = self.input
        # if input is not None:
        #     if not input.shape:
        #         input = input.expand((*self.batch_shape, self.in_cells))
        #     else:
        #         input = input.reshape((*self.batch_shape, -1))
        #     input = input.float()
        # return m(input).reshape(self.shape)
        return m(self.input)
