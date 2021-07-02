from typing import cast

import torch
import torch_semiring_einsum as tse
import torchfactors  # type: ignore
from torchfactors.einsum import (MultiEquation, compile_equation,
                                 compile_obj_equation, log_einsum, my_align_to)


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


def test_compile_obj_equation_multi1():
    from_str, = cast(MultiEquation, compile_equation('ij,jk,kl->jl', force_multi=True)).equations
    i, j, k, el = [object() for _ in range(4)]
    from_obj, = cast(MultiEquation, compile_obj_equation(
        [[i, j], [j, k], [k, el]],
        [[j, el]],
        force_multi=True)).equations
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


def test_log_einsum():
    from_str = cast(tse.equation.Equation, compile_equation('ij,jk,kl->jl'))
    a, b, c = [torch.zeros(3, 3)] * 3
    jl = log_einsum(from_str, a, b, c)
    assert (jl == torch.tensor(9.0).log()).all()


def test_log_einsum_multi():
    from_str = cast(tse.equation.Equation, compile_equation('ij,jk,kl->jl', force_multi=True))
    a, b, c = [torch.zeros(3, 3)] * 3
    jl, = log_einsum(from_str, a, b, c)
    assert (jl == torch.tensor(9.0).log()).all()


def test_align_to():
    ab = torch.tensor([
        [1, 2, 3],
        [2, 4, 5]
    ])
    out = my_align_to(ab, [89, 2], [9, 2, 89, 345])
    assert out.shape == (1, 3, 2, 1)


def test_log_dot():
    ab = torch.tensor([
        [1, 2, 3],
        [2, 4, 5]
    ])
    bc = torch.tensor([
        [1, 2, 3, 8],
        [2, 4, 5, 3],
        [1, 4, 5, 3],
    ])
    ac_expected = torch.tensor([  # 2 x 4
        [1+4+3, 2+8+12, 3+10+15, 8+6+9],
        [2+8+5, 4+16+20, 6+20+25, 16+12+15]
    ]).float()
    log_out, = torchfactors.log_dot(
        [(ab.log(), list('ab')),
         (bc.log(), list('bc'))],
        list('ac'))
    out = log_out.exp()
    assert out.allclose(ac_expected)


# print(_log_dot.code)
# print(my_align_to.code)
