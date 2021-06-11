
from functools import singledispatch
from typing import Dict, Hashable, List, Sequence, Tuple, Union

import torch_semiring_einsum as tse  # type: ignore
from torch import Tensor


class MultiEquation(object):
    def __init__(self, equations: Sequence[tse.equation.Equation]):
        self.equations = list(equations)


@singledispatch
def log_einsum(compiled_equation: tse.equation.Equation, *tensors: Tensor):
    return tse.log_einsum(compiled_equation, tensors)


@log_einsum.register
def _from_multi(compiled_equation: MultiEquation, *tensors: Tensor):
    return tuple(tse.log_einsum(eq, tensors) for eq in compiled_equation.equations)


def compile_equation(equation: str, force_multi: bool = False
                     ) -> tse.equation.Equation:
    args_str, outputs_str = equation.split('->', 1)
    arg_strs = args_str.split(',')
    out_strs = outputs_str.split(',')
    return compile_generic_equation(arg_strs, out_strs, repr=equation, force_multi=force_multi)


def compile_generic_equation(arg_strs: Sequence[Sequence[Hashable]],
                             out_strs: Sequence[Sequence[Hashable]],
                             repr: str = '', force_multi: bool = False
                             ) -> Union[tse.equation.Equation, MultiEquation]:
    r"""modified from: https://github.com/bdusell/semiring-einsum/blob/7fbebdddc70aab81ede5e7c086719bff700b3936/torch_semiring_einsum/equation.py#L63-L92

    Pre-compile an einsum equation for use with the einsum functions in
    this package.

    :return: A pre-compiled equation.
    """  # noqa: E501
    char_to_int: Dict[Hashable, int] = {}
    int_to_arg_dims: List[List[Tuple[int, int]]] = []
    args_dims: List[List[int]] = []
    for arg_no, arg_str in enumerate(arg_strs):
        arg_dims = []
        for dim_no, dim_char in enumerate(arg_str):
            dim_int = char_to_int.get(dim_char)
            if dim_int is None:
                dim_int = char_to_int[dim_char] = len(char_to_int)
                int_to_arg_dims.append([])
            int_to_arg_dims[dim_int].append((arg_no, dim_no))
            arg_dims.append(dim_int)
        args_dims.append(arg_dims)
    num_variables = len(char_to_int)
    equations = []
    for out_str in out_strs:
        output_dims = [char_to_int[c] for c in out_str]
        equations.append(tse.equation.Equation(
            repr,
            int_to_arg_dims,
            args_dims,
            output_dims,
            num_variables))
    if len(equations) != 1 or force_multi:
        return MultiEquation(equations)
    else:
        return equations[0]
