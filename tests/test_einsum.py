from typing import cast

import torch_semiring_einsum as tse  # type: ignore
from torchfactors.einsum import compile_equation, compile_obj_equation


def test_compile_obj_equation():
    from_str = cast(tse.equation.Equation, compile_equation('ij,jk,kl->jl'))
    i, j, k, el = [object() for _ in range(4)]
    from_obj = cast(tse.equation.Equation, compile_obj_equation(
        [[i, j], [j, k], [k, el]],
        [[j, el]]))
    assert from_str.input_variables == [
        [0, 1], [1, 2], [2, 3]
    ]
    assert from_str.output_variables == [1, 3]
    # first var abppear in first input at position 0
    assert from_str.variable_locations == [
        [(0, 0)],
        [(0, 1), (1, 0)],
        [(1, 1), (2, 0)],
        [(2, 1)]
    ]
    assert from_str.input_variables == from_obj.input_variables
    assert from_str.output_variables == from_obj.output_variables
    assert from_str.variable_locations == from_obj.variable_locations
