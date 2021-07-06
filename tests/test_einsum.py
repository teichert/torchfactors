from typing import List, cast

import torch
import torch_semiring_einsum as tse  # type: ignore
import torchfactors as tx
from torchfactors.einsum import (MultiEquation, compile_equation,
                                 compile_obj_equation, log_einsum)


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
    log_out, = tx.log_dot(
        [(ab.log(), tx.ids('ab')),
         (bc.log(), tx.ids('bc'))],
        [tx.ids('ac')])
    out = log_out.exp()
    assert out.allclose(ac_expected)


def test_hmm():
    t = torch.tensor([
        [-7.021953582763672, 3.808997869491577],
        [4.331912040710449, -17.0089054107666]
    ])
    # t = torch.tensor([
    #     [0.5, 1],
    #     [1, 0.5]
    # ])
    print(t)
    # a b c
    # x y z
    # out_free = torch.einsum('ax,by,ab,cz,bc->', t, t, t, t, t)

    out_is_True = torch.tensor([[False, True], [False, True]])
    ax = t.masked_fill(out_is_True, float('-inf'))
    by = t.masked_fill(out_is_True, float('-inf'))
    cz = t.masked_fill(out_is_True.logical_not(), float('-inf'))

    out_free, = tx.log_dot(
        [
            (t, tx.ids('ax')),
            (t, tx.ids('bx')),
            (t, tx.ids('cz')),
            (t, tx.ids('ab')),
            (t, tx.ids('bc'))],
        [[]])

    out_clamped, = tx.log_dot(
        [
            (ax, tx.ids('ax')),
            (by, tx.ids('bx')),
            (cz, tx.ids('cz')),
            (t, tx.ids('ab')),
            (t, tx.ids('bc'))],
        [[]])

    a, b, c = [tx.TensorVar(tx.Range(2), tensor=torch.tensor(0), usage=tx.LATENT)
               for _ in range(3)]
    x, y = [tx.TensorVar(tx.Range(2), tensor=torch.tensor(0), usage=tx.ANNOTATED)
            for _ in range(2)]
    z = tx.TensorVar(tx.Range(2), tensor=torch.tensor(1), usage=tx.ANNOTATED)

    variables: List[tx.Var] = [a, b, c, x, y, z]
    f_ax = tx.TensorFactor(a, x, tensor=t)
    f_by = tx.TensorFactor(b, y, tensor=t)
    f_cz = tx.TensorFactor(c, z, tensor=t)
    f_ab = tx.TensorFactor(a, b, tensor=t)
    f_bc = tx.TensorFactor(b, c, tensor=t)
    factors = [
        f_ax,
        f_by,
        f_cz,
        f_ab,
        f_bc,
    ]
    bp = tx.BP()
    for v in variables:
        v.clamp_annotated()

    logz_clamped = bp.product_marginal(factors)
    print([f.dense for f in factors])

    tx.factor.uncache(factors)
    for v in variables:
        v.unclamp_annotated()

    logz_free = bp.product_marginal(factors)
    print([f.dense for f in factors])
    print(out_free, out_clamped)
    print(logz_free, logz_clamped)
    assert logz_free.allclose(out_free)
    assert logz_clamped.allclose(out_clamped)


test_hmm()
