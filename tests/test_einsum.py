from typing import List, cast

import torch
import torch_semiring_einsum as tse  # type: ignore
import torchfactors as tx
from torchfactors.einsum import (MultiEquation, compile_equation,
                                 compile_obj_equation, log_einsum,
                                 nonan_logsumexp)


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


def test_expand_to():
    t = torch.ones(3, 5)
    shape = [2, 2, 3, 2, 5, 2]
    out = tx.einsum.expand_to(t, [2, 4], shape)
    assert (out == torch.ones(shape)).all()


# def test_map_and_order_tensor():
#     t1 = torch.ones(3, 4, 5, 2)
#     t2 = torch.ones(2, 7, 3)
#     t3 = torch.ones(3)
#     vocab: Dict[str, Tuple[int, int]] = {}
#     out_t1, out_names1 = map_and_order_tensor(t1, vocab, ['19', '10', '11', '13'])
#     assert out_t1.shape == (3, 4, 5, 2)
#     assert out_names1 == [0, 1, 2, 3]

#     out_t2, out_names2 = map_and_order_tensor(t2, vocab, ['13', '40', '19'])
#     assert out_t2.shape == (3, 2, 7)
#     assert out_names2 == [0, 3, 4]

#     out_t3, out_names3 = map_and_order_tensor(t3, vocab, ['00'])
#     assert out_t3.shape == (3,)
#     assert out_names3 == [5]

#     assert vocab == {
#         '19': (0, 3),
#         '10': (1, 4),
#         '11': (2, 5),
#         '13': (3, 2),
#         '40': (4, 7),
#         '00': (5, 3),
#     }

#     q1 = ['13', '19', '00', '11']
#     out_q1, permutation_q1 = map_and_order_names(vocab, q1)
#     assert out_q1 == [0, 2, 3, 5]  # sorted([3, 0, 5, 2])
#     assert permutation_q1 == [2, 0, 3, 1]

#     q2 = ['00', '10', '13', '11']
#     out_q2, permutation_q2 = map_and_order_names(vocab, q2)
#     assert out_q2 == [1, 2, 3, 5]  # sorted([5, 1, 3, 2])
#     # 0 1 2 3 # indexes
#     # 5 1 3 2 # orig ids
#     # so in order of smallest orig id, the indexes become:
#     # 1 3 2 0 # indexes in orig list to get sorted ids (sorted indexes)
#     # 1 2 3 5 # sorted ids
#     # so in order of sorted index, the index become
#     # 3 0 2 1
#     assert permutation_q2 == [3, 0, 2, 1]

#     ordered_dims_to_sum, fix_permutation = map_order_and_invert_query(vocab, q1)
#     assert ordered_dims_to_sum == [1, 4]
#     assert fix_permutation == [2, 0, 3, 1]


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
        [(ab.log(), list('ab')),
         (bc.log(), list('bc'))],
        [list('ac')])
    out = log_out.exp()
    assert out.allclose(ac_expected)


test_log_dot()

# def test_log_dot1_5():
#     t1 = torch.eye(2).log()
#     t2 = torch.tensor([0, 1])
#     out, = tx.log_dot([
#         (t1, [57536, 57248]),
#         (t2, [65424]),
#     ], [[65424], [57248]])


def test_log_dot2():
    t = torch.tensor([[1, 2], [0.5, 3]]).float().log()
    out, = tx.log_dot([
        (t, list('ab'))
    ], [[]])
    assert out == t.logsumexp(dim=[0, 1])


def test_hmm():
    # t = torch.tensor([
    #     [-7.021953582763672, 3.808997869491577],
    #     [4.331912040710449, -17.0089054107666]
    # ])
    t = torch.tensor([
        [1., 2.],
        [3., 4.]
    ])
    print(t)

    out_is_True = torch.tensor([[False, True], [False, True]])
    ax = t.masked_fill(out_is_True, float('-inf'))
    by = t.masked_fill(out_is_True, float('-inf'))
    cz = t.masked_fill(out_is_True.logical_not(), float('-inf'))

    for q in ['', 'ax', 'by', 'cz', 'ab', 'bc', 'a', 'b', 'c', 'x', 'y', 'z']:
        gold_clamped = torch.einsum(f'ax,by,cz,ab,bc->{q}', ax.exp(), by.exp(),
                                    cz.exp(), t.exp(), t.exp()).log()
        print(f'gold clamped "{q}": {tx.Factor.normalize(gold_clamped).tolist()}')
    gold_free = torch.einsum('ax,by,cz,ab,bc->', [t.exp()] * 5).log()
    print(f'gold free: {gold_free}')

    free_factors = [
        (t, list('ax')),
        (t, list('by')),
        (t, list('cz')),
        (t, list('ab')),
        (t, list('bc'))]
    out_free, = tx.log_dot(free_factors, [[]])

    clamped_factors = [
        (ax, list('ax')),
        (by, list('by')),
        (cz, list('cz')),
        (t, list('ab')),
        (t, list('bc')),
    ]
    out_clamped, = tx.log_dot(clamped_factors, [[]])

    a, b, c = [tx.TensorVar(tx.Range(2), tensor=torch.tensor(0), usage=tx.LATENT, info=name)
               for name in 'abc']
    x, y = [tx.TensorVar(tx.Range(2), tensor=torch.tensor(0), usage=tx.ANNOTATED, info=name)
            for name in 'xy']
    z = tx.TensorVar(tx.Range(2), tensor=torch.tensor(1), usage=tx.ANNOTATED, info='z')

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


def test_nonan_logsumexp():
    out = nonan_logsumexp(torch.ones(3, 4), dims=[])
    assert out.allclose(torch.ones(3, 4))
