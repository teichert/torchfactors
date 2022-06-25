from __future__ import annotations

import math
from typing import Optional, Union, cast

import torch
from torch import Tensor

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
                 bias: bool = True, device=None, dtype=None,
                 fix_last: Optional[float] = None):
        factory_kwargs = dict(device=device, dtype=dtype)
        super().__init__()
        # self.initialized = False
        self.input_features = input_features
        self.output_features = output_features
        self.bias = bias
        self.bias_only: Union[torch.nn.parameter.Parameter, torch.Tensor, None] = None
        self.with_features: Optional[torch.nn.Module] = None
        self.fix_last = None if fix_last is None else torch.tensor([fix_last], dtype=torch.float)
        # def lazy_init(self, exemplar: Tensor):
        # if not self.initialized:
        if self.fix_last is None:
            effective_output_features = self.output_features
        else:
            effective_output_features = self.output_features - 1
        if self.input_features == 0:
            if self.bias:
                new_bias = torch.empty(effective_output_features,
                                       **factory_kwargs)
                torch.nn.init.uniform_(new_bias, -1., 1.)
                self.bias_only = torch.nn.parameter.Parameter(new_bias)
            else:
                self.bias_only = torch.tensor(0.0, **factory_kwargs).expand(
                    (effective_output_features,))
        else:
            self.with_features = torch.nn.Linear(
                self.input_features, effective_output_features, bias=self.bias)
        # self.initialized = True

    def forward(self, x: Tensor) -> Tensor:
        # self.lazy_init(x)
        if self.bias_only is not None:
            # out = self.bias_only.expand(*x.shape[:-1], len(self.bias_only))
            out = self.bias_only
        else:
            out = cast(torch.nn.Module, self.with_features)(x)
        if self.fix_last is not None:
            out = torch.cat([out, self.fix_last], dim=-1)
        return out

# def register_module_for_state_dict(cls):
#     register_module_for_state_dict.known_classes

# register_module_for_state_dict.known_classes = {}
# @register_module_for_state_dict


class ShapedLinear(torch.nn.Module):
    r"""
    Like a built-in Linear layer, but the output
    is automatically reshaped and may be bias-only
    """

    def __init__(self, output_shape: ShapeType,
                 bias: bool = True, input_shape: ShapeType = None,
                 fix_last: Optional[float] = None):
        super().__init__()
        self.input_shape = input_shape
        if input_shape is None:
            input_features = 0
        else:
            input_features = math.prod(input_shape)
        self.output_shape = output_shape
        output_features = math.prod(output_shape)
        self.wrapped_linear = OptionalBiasLinear(
            input_features, output_features,
            bias=bias, fix_last=fix_last)

    def forward(self, x: Tensor) -> Tensor:
        flattened_in: Tensor
        if self.input_shape is None or not self.input_shape:
            flattened_in = x
        else:
            flattened_in = x.flatten(len(x.shape) - len(self.input_shape))
        out: Tensor = self.wrapped_linear(flattened_in)
        reshaped = out.unflatten(-1, self.output_shape)
        return reshaped


def inner_shape(shape: ShapeType) -> ShapeType:
    return tuple(s - 1 for s in shape)


class MinimalLinear(torch.nn.Module):

    def __init__(self, output_shape: ShapeType,
                 bias: bool = True, input_shape: ShapeType = None,
                 fix_last: Optional[float] = None):
        super().__init__()
        self.output_shape = output_shape
        self.inner = ShapedLinear(inner_shape(output_shape), bias=bias, input_shape=input_shape,
                                  fix_last=fix_last)

    def forward(self, x: Tensor) -> Tensor:
        t = self.inner(x)
        out_dims = len(self.output_shape)
        t_dims = len(t.shape)
        # for all non-batch dims, extend left with zeros
        for dim in range(t_dims - out_dims, t_dims):
            t = torch.cat((torch.zeros_like(t.select(dim, 0)).unsqueeze(dim), t), dim=dim)
        return t
        # flattened_in: Tensor
        # if self.input_shape is None or not self.input_shape:
        #     flattened_in = x
        # else:
        #     flattened_in = x.flatten(len(x.shape) - len(self.input_shape))
        # out: Tensor = self.wrapped_linear(flattened_in)
        # reshaped = out.unflatten(-1, self.output_shape)
        # for d in range(len(self.output_shape)):
        #     rest = len(self.output_shape) - d - 1
        #     reshaped[(None,)*d + (0,) + (None,) * rest] = 0.0
        # return reshaped


def LinearTensor(params: ParamNamespace,
                 *variables: Var,
                 bias: bool = True,
                 minimal: bool = False,
                 fix_last: Optional[float] = None):
    return LinearTensorAux(params, *variables, out_shape=Factor.out_shape_from_variables(variables),
                           bias=bias, minimal=minimal, fix_last=fix_last)


def LinearTensorAux(params: ParamNamespace,
                    *variables_for_in_shape: Var,
                    out_shape: ShapeType,
                    bias: bool = True,
                    minimal: bool = False,
                    fix_last: Optional[float] = None):
    def f(input: Optional[Tensor]) -> Tensor:
        batch_shape = Factor.batch_shape_from_variables(variables_for_in_shape)
        in_shape = None if input is None else input.shape[len(batch_shape):]
        m = params.module(
            (MinimalLinear if minimal else ShapedLinear), output_shape=out_shape,
            input_shape=in_shape, bias=bias, fix_last=fix_last)
        if input is None:
            x = variables_for_in_shape[0].tensor.new_empty(0, dtype=torch.float)
        else:
            x = input
        return m(x)
    return f


class LinearFactor(Factor):
    r"""
    A factor for which the configuration scores come from a
    ShapedLinear layer applied to the specified input:

    the input should exactly match all batch dim (including graph_dims)
    the remaining dimensions will be used to get a different
    output for each batch element
    """

    def __init__(self,
                 params: ParamNamespace,
                 *variables: Var,
                 input: Optional[Tensor] = None,
                 bias: bool = True,
                 share: bool = False,
                 minimal: bool = False):
        r"""
        share: if True, then the input will be expanded to match the graph_dims
        of the first variable (i.e. using the same features within a particular
        batch element)

        TODO: this is hard for me to make sense of. share=True seems to do the
        opposite of what the name indicates.  Rather than having the same
        parameters reused across the batch elements, it looks like share=True
        increases the number of parameters by having the extra dimensions
        represents additional separate outputs that each get their own separate
        multiplicative parameters, but the bias is still shared across; this
        seems like a bad name at best
        """
        super().__init__(variables)
        self.minimal = minimal
        self.bias = bias
        self.params = params
        self.get_tensor = LinearTensor(params, *variables, minimal=minimal, bias=bias)
        if input is not None and input.shape:
            if share:
                # repeat the input across each replicate in the batch element graph
                graph_dims = variables[0].ndims
                actual_batch_dims = self.num_batch_dims - graph_dims
                actual_batch_shape = input.shape[:actual_batch_dims]
                input_shape = input.shape[actual_batch_dims:]
                if actual_batch_shape == self.batch_shape[:actual_batch_dims]:
                    graph_shape = variables[0].shape[actual_batch_dims:]
                    # leave batch_dims alone, add graph_dims
                    input = input[(slice(None),) * actual_batch_dims + (None,) * variables[0].ndims]
                    input = input.expand((-1,) * actual_batch_dims + graph_shape + input_shape)
                else:
                    raise ValueError("tried to expand shared input, but it didn't work")
            if input.shape[:self.num_batch_dims] != self.batch_shape:
                raise ValueError("prefix dimensions of input must match batch_dims")
        self.input = input

    # @property
    # def in_shape(self) -> Optional[ShapeType]:
    #     if self.input is None:
    #         return None
    #     else:
    #         return self.input.shape[len(self.batch_shape):]

    def dense_(self) -> torch.Tensor:
        r"""returns a tensor that characterizes this factor;

        the factor's variable-domains dictate the number and
        size of the final dimensions.
        the variables themselves, then, know how many batch
        dimensions there are.
        """
        # m = self.params.module(
        #     (MinimalLinear if self.minimal else ShapedLinear), output_shape=self.out_shape,
        #     input_shape=self.in_shape, bias=self.bias)
        # if self.input is None:
        #     x = self.variables[0].tensor.new_empty(0, dtype=torch.float)
        # else:
        #     x = self.input
        # return m(x)
        return self.get_tensor(self.input)
