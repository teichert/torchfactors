import pytest
import torch
from torchfactors import Range, TensorFactor, TensorVar, ndrange
from torchfactors.factor import Factor


def test_factor():
    t = torch.ones(3, 4)
    v = TensorVar(t, domain=Range(10))
    factor_tensor = ndrange(3, 4, 10).log()
    f: Factor = TensorFactor(v, factor_tensor)
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
    # f.tensor = torch.arange(3, 4, 10)
    # f.normalize()
    # assert f.tensor.exp().sum() == 1.0


def test_bad_tensor_factor():
    t = torch.ones(3, 4)
    v = TensorVar(t, domain=Range(10))
    with pytest.raises(ValueError):
        TensorFactor(v, torch.rand(3, 4, 9))


def test_init_tensor_factor():
    t = torch.ones(3, 4)
    v = TensorVar(t, domain=Range(10))
    f = TensorFactor(v)
    assert f.dense.shape == (3, 4, 10)
    assert (f.dense == 0).all()
