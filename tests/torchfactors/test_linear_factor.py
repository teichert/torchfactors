
from typing import Any

import pytest
import torch

from torchfactors import Range, TensorVar
from torchfactors.components.linear_factor import LinearFactor, MinimalLinear
from torchfactors.model import Model
from torchfactors.utils import num_trainable


def test_linear_factor():
    m = Model[Any]()
    v = TensorVar(Range(10), torch.zeros(10))
    f = LinearFactor(m.namespace('root'), v)
    print(f.dense)
    p = list(m.parameters())
    print(p)
    assert len(p) == 1


def test_linear_factor2():
    m = Model[Any]()
    v = TensorVar(Range(10), torch.zeros(10))
    f = LinearFactor(m.namespace('root'), v, input=torch.ones(10, 7, 13, 10))
    print(f.dense)
    p = list(m.parameters())
    print(p)
    assert len(p) == 2


def test_linear_factor2b():
    m = Model[Any]()
    v = TensorVar(Range(10), torch.zeros(3))
    f = LinearFactor(m.namespace('root'), v, input=torch.ones(7, 13, 8), share=True)
    print(f.dense)
    p = list(m.parameters())
    print(p)
    assert len(p) == 2
    # This really doesn't make sense to me, the three separate variables all use the same inputs in the same way
    expected_params = 10 * 7 * 13 * 8 + 10
    out_params = num_trainable(m)
    assert out_params == expected_params


def test_linear_factor2c():
    m = Model[Any]()
    v = TensorVar(Range(10), torch.zeros(3))
    f = LinearFactor(m.namespace('root'), v, input=torch.ones(3, 7, 13))
    print(f.dense)
    p = list(m.parameters())
    print(p)
    assert len(p) == 2
    expected_params = 10 * 7 * 13 + 10
    out_params = num_trainable(m)
    assert out_params == expected_params


def test_linear_factor2d():
    m = Model[Any]()
    v = TensorVar(Range(10), torch.zeros(3, 13))
    f = LinearFactor(m.namespace('root'), v, input=torch.ones(3, 13, 7))
    print(f.dense)
    p = list(m.parameters())
    print(p)
    assert len(p) == 2
    expected_params = 10 * 7 + 10
    out_params = num_trainable(m)
    assert out_params == expected_params


def test_bad_linear_factor():
    m = Model[Any]()
    v = TensorVar(Range(10), torch.zeros(10))
    with pytest.raises(ValueError):
        LinearFactor(m.namespace('root'), v, input=torch.ones(13, 10))


def test_no_bias_no_input():
    m = Model[Any]()
    v = TensorVar(Range(7), torch.ones(10))
    f = LinearFactor(m.namespace('root'), v, bias=False)
    out = f.dense
    expected = torch.zeros(10, 7)
    assert out.allclose(expected)


def test_minimal_linear():
    m = Model[Any]()
    v = TensorVar(Range(10), torch.tensor(0.0))
    f = LinearFactor(m.namespace('root'), v, input=torch.ones(7), minimal=True)
    print(f.dense)
    p = list(m.parameters())
    print(p)
    assert len(p) == 2
    expected_params = 7 * 9 + 9
    out_params = num_trainable(m)
    assert out_params == expected_params


def test_minimal_linear_with_batch1():
    m = Model[Any]()
    v = TensorVar(Range(10), torch.zeros(3, 13))
    f = LinearFactor(m.namespace('root'), v, input=torch.ones(3, 13, 7), minimal=True)
    print(f.dense)
    p = list(m.parameters())
    print(p)
    assert len(p) == 2
    expected_params = 9 * 7 + 9
    out_params = num_trainable(m)
    assert out_params == expected_params


def test_minimal_linear_with_batch2():
    m = Model[Any]()
    v = TensorVar(Range(10), torch.zeros(3))
    f = LinearFactor(m.namespace('root'), v, input=torch.ones(3, 7, 13), minimal=True)
    print(f.dense)
    p = list(m.parameters())
    print(p)
    assert len(p) == 2
    expected_params = 9 * 7 * 13 + 9
    out_params = num_trainable(m)
    assert out_params == expected_params


# def test_linear_factor_with_single_dim_input():
#     m = Model[Any]()
#     n = 7
#     v = TensorVar(Range(10), torch.zeros(n))
#     f = LinearFactor(m.namespace('root'), v, input=torch.ones(n))
#     assert f.dense.shape == (10, 1)
