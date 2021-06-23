import math

import pytest
import torch
from torchfactors import Range, TensorFactor, TensorVar, ndarange
from torchfactors.factor import Factor


def test_factor():
    t = torch.ones(3, 4)
    v = TensorVar(t, domain=Range(10))
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

# TODO: test where they vary accross batch dims


def test_bad_variables():
    # cannot pass more than one variable without putting them into a sequence
    v1 = TensorVar(torch.ones(3, 4), domain=Range(10))
    v2 = TensorVar(torch.ones(3, 4), domain=Range(5))
    t = TensorFactor(v1, v2)
    assert t.shape == (3, 4, 10, 5)
