
from typing import Any

import pytest
import torch
from torchfactors import Range, TensorVar
from torchfactors.components.linear_factor import LinearFactor
from torchfactors.model import Model


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
