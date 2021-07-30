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
        self.initialized = False
        self.input_features = input_features
        self.output_features = output_features
        self.bias = bias
        self.bias_only: Union[torch.nn.parameter.Parameter, torch.Tensor, None] = None
        self.with_features: Optional[torch.nn.Module] = None

    def lazy_init(self, exemplar: Tensor):
        if not self.initialized:
            if self.input_features == 0:
                if self.bias:
                    bias = exemplar.new_empty(self.output_features)
                    torch.nn.init.uniform_(bias, -1., 1.)
                    self.bias_only = torch.nn.parameter.Parameter(bias)
                else:
                    self.bias_only = exemplar.new_tensor(0.0).expand((self.output_features,))
            else:
                self.with_features = torch.nn.Linear(
                    self.input_features, self.output_features, bias=self.bias)
            self.initialized = True

    def forward(self, x: Tensor) -> Tensor:
        self.lazy_init(x)
        if self.bias_only is not None:
            # out = self.bias_only.expand(*x.shape[:-1], len(self.bias_only))
            out = self.bias_only
        else:
            out = cast(torch.nn.Module, self.with_features)(x)
        return out


class ShapedLinear(torch.nn.Module):
    r"""
    Like a built-in Linear layer, but the output
    is automatically reshaped and may be bias-only
    """

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
    r"""
    A factor for which the configuration scores come from a
    ShapedLinear layer applied to the specified input
    """

    def __init__(self,
                 params: ParamNamespace,
                 *variables: Var,
                 input: Optional[Tensor] = None,
                 bias: bool = True,
                 graph_dims: int = 0):
        super().__init__(variables, graph_dims=graph_dims)
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
        if self.input is None:
            x = self.variables[0].tensor.new_empty(0, dtype=torch.float)
        else:
            x = self.input
        return m(x)
