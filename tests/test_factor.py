import math

import pytest
import torch
from torchfactors import (ANNOTATED, CLAMPED, LATENT, OBSERVED, PADDING,
                          Factor, Range, TensorFactor, TensorVar, ndarange)


def test_factor():
    t = torch.ones(3, 4)
    v = TensorVar(t, Range(10), LATENT)
    factor_tensor = ndarange(3, 4, 10).log()
    with pytest.raises(TypeError):
        # needs to name factor_tensor argument since it comes after varargs
        TensorFactor(v, factor_tensor)  # type: ignore
    f: Factor = TensorFactor(v, tensor=factor_tensor)
    assert f.shape == (3, 4, 10)
    assert f.batch_cells == 3 * 4
    assert f.batch_shape == (3, 4)
    assert f.num_batch_dims == 2
    assert f.out_shape == (10,)
    assert f.cells == 3 * 4 * 10
    assert f.free_energy
    assert len(f) == 1
    assert list(f) == [v]
    zs = f.product_marginal()
    expected_zs = factor_tensor.logsumexp(dim=-1)
    assert zs.isclose(expected_zs).all()
    zs2 = -f.free_energy()
    assert zs2.isclose(expected_zs).all()


def test_bad_tensor_factor():
    t = torch.ones(3, 4)
    v = TensorVar(t, domain=Range(10))
    with pytest.raises(ValueError):
        TensorFactor(v, tensor=torch.rand(3, 4, 9))


def test_bad_repeat_var():
    t = torch.ones(3, 4)
    v = TensorVar(t, domain=Range(10))
    with pytest.raises(ValueError):
        TensorFactor(v, v, tensor=torch.rand(3, 4, 10, 10))


def test_init_tensor_factor():
    t = torch.ones(3, 4)
    v = TensorVar(t, domain=Range(10))
    f = TensorFactor(v)
    assert f.dense.shape == (3, 4, 10)
    assert (f.dense == 0).all()


def test_multi_factor():
    v1 = TensorVar(torch.ones(3, 4), domain=Range(10))
    v2 = TensorVar(torch.ones(3, 4), domain=Range(5))
    f = TensorFactor(v1, v2)
    assert f.shape == (3, 4, 10, 5)


def test_bad_multi_factor():
    v1 = TensorVar(torch.ones(3, 4), domain=Range(10))
    v2 = TensorVar(torch.ones(4, 4), domain=Range(10))
    with pytest.raises(ValueError):
        # batch dims don't match
        TensorFactor(v1, v2)


def test_multi_factor_var():
    v1 = TensorVar(torch.ones(3, 4), domain=Range(10))
    v2 = TensorVar(torch.ones(3, 4), domain=Range(5))
    f = TensorFactor(v1, v2)
    assert f.product_marginal(v1).shape == (3, 4, 10)
    assert (f.product_marginal(v1).exp() == 5).all()
    assert f.product_marginal(v2).shape == (3, 4, 5)
    assert (f.product_marginal(v2).exp() == 10).all()


def test_multi_factor_2vars():
    v1 = TensorVar(torch.ones(3, 4), domain=Range(10))
    v2 = TensorVar(torch.ones(3, 4), domain=Range(5))
    f = TensorFactor(v1, v2)
    v1marg, v2marg = f.product_marginals([v1], [v2])
    assert v1marg.shape == (3, 4, 10)
    assert (v1marg.exp() == 5).all()
    assert v2marg.shape == (3, 4, 5)
    assert (v2marg.exp() == 10).all()


def test_multi_factor_bad_2vars():
    v1 = TensorVar(torch.ones(3, 4), domain=Range(10))
    v2 = TensorVar(torch.ones(3, 4), domain=Range(5))
    f = TensorFactor(v1, v2)
    with pytest.raises(ValueError):
        # missing the brackets around the variables makes things ambiguous
        f.product_marginals(v1, v2)  # type: ignore


def test_normalize():
    out = Factor.normalize(torch.ones(2, 3, 4, 5), 2)
    expected = torch.full((2, 3, 4, 5), fill_value=-math.log(20))
    assert out.isclose(expected).all()


# # TODO: test where they vary accross batch dims
# def test_multiple_graph_dims():
#     v = TensorVar(torch.ones(5, 4, 6), domain=Range(2), usage=tx.ANNOTATED)
#     # 5 per batch
#     # (4-1) x 6 = 18 factors per graph
#     f = TensorFactor(v[..., :-1, :], v[..., 1:, :], tensor=torch.tensor([
#         [1, 0],
#         [0, 1],
#     ]).log(), graph_dims=2)
#     assert f.shape == (5, 3, 6, 2, 2)
#     assert f.input_shape == (5, 3, 6)
#     assert f.out_shape == (2, 2)
#     assert f.batch_shape == (5,)

#     # a0 a1
#     # b0 b1
#     # c0 c1
#     # equivalent to separately adding in:
#     # TF(a0, b0)
#     # TF(b0, c0)
#     # TF(c0, d0)
#     #  ...
#     # TF(a5, b5)
#     # TF(b5, c5)
#     # TF(c5, d5)
#     # the factor says that these pairs need to be the same
#     # so the 0s and 1s can each choose separately 0 or 1;
#     # if I clamp one of the 1's at the end, all of the 1's should get set to True
#     # represents one out of 2 ** 5 == 32 legal configurations (for each batch)
#     v[..., 0, 0].set_usage(tx.CLAMPED)
#     # v[..., 0, 0].set_tensor(0)
#     # tx.BP(passes=10).predict([f])
#     # out = v[..., 0].tensor
#     # assert (out == 0).all()
#     v[..., 0, 0].set_tensor(1)
#     tx.BP(passes=10).predict([f])
#     out = v[..., 0].tensor
#     assert (out == 1).all()


# test_multiple_graph_dims()


def test_bad_variables():
    # cannot pass more than one variable without putting them into a sequence
    v1 = TensorVar(torch.ones(3, 4), domain=Range(10))
    v2 = TensorVar(torch.ones(3, 4), domain=Range(5))
    t = TensorFactor(v1, v2)
    assert t.shape == (3, 4, 10, 5)


def test_mask():
    v = TensorVar(tensor=torch.tensor([
        2,
        3,
        1,
        2,
        1,
    ]), domain=Range(5), usage=torch.tensor([
        ANNOTATED,
        OBSERVED,
        LATENT,
        CLAMPED,
        PADDING,
    ]))
    factor = TensorFactor(v, tensor=torch.full(v.marginal_shape, math.log(3)))
    out = factor.dense
    assert out.allclose(torch.tensor([
        [3, 3, 3, 3, 3],
        [0, 0, 0, 3, 0],
        [3, 3, 3, 3, 3],
        [0, 0, 3, 0, 0],
        [0, 1, 0, 0, 0],
    ]).log())


def test_mask2():
    v = TensorVar(tensor=torch.tensor([
        1,
        0,
        1,
        1,
        0,
    ]), domain=Range(2), usage=torch.tensor([
        PADDING,
        CLAMPED,
        ANNOTATED,
        OBSERVED,
        LATENT,
    ]))

    v2 = TensorVar(tensor=torch.tensor([
        2,
        1,
        2,
        3,
        1,
    ]), domain=Range(4), usage=torch.tensor([
        ANNOTATED,
        OBSERVED,
        LATENT,
        PADDING,
        CLAMPED,
    ]))

    factor = TensorFactor(v, v2, tensor=torch.full((5, 2, 4), math.log(3)))
    out = factor.dense  # in x v x v2
    assert out.allclose(torch.tensor([
        [  # P x A
            [0, 0, 0, 0],
            [1, 1, 1, 1],
        ],
        [  # C x O
            [0, 3, 0, 0],
            [0, 0, 0, 0],
        ],
        [  # A x L
            [3, 3, 3, 3],
            [3, 3, 3, 3],
        ],
        [   # O x P
            [0, 0, 0, 0],
            [0, 0, 0, 1],
        ],
        [  # L x C
            [0, 3, 0, 0],
            [0, 3, 0, 0],
        ],
    ]).log())
