from typing import Dict, Tuple

import pytest
import torch
from torchfactors import (ANNOTATED, CLAMPED, DEFAULT, LATENT, OBSERVED,
                          VarBase, VarUsage)
from torchfactors.domain import Range
from torchfactors.variable import Var, VarField, compose, compose_single


def test_usage1():
    assert VarUsage.DEFAULT == VarUsage.OBSERVED


def test_variable():
    t = torch.ones(3, 4)
    v = Var(t, Range[4])
    assert len(v.domain) == 4
    assert v.tensor is t
    assert v.shape == (3, 4)
    v.set_usage(OBSERVED)
    assert isinstance(v.usage, torch.Tensor)
    print(v.usage == VarUsage.OBSERVED)
    assert (v.usage == VarUsage.OBSERVED).all()
    assert v.ndslice == (...,)
    assert v.original_tensor is t


def test_shape():
    t = torch.ones(3, 4)
    v = Var(t, Range[4])
    v2 = v[1:3, 2:4]
    assert v2.original_tensor is t
    assert v2.ndslice == (slice(1, 3), slice(2, 4))
    assert t.sum() == 3*4
    assert v2.shape == (2, 2)
    # could have done v2.tensor = 0.0, but waiting on:
    # https://github.com/python/mypy/issues/3004
    v2.set_tensor(0.0)
    assert v2.tensor.sum() == 0.0
    assert t.sum() == 3*4 - 4


def test_usage3():
    t = torch.ones(3, 4)
    v = Var(t, Range[4])
    v.set_usage(ANNOTATED)
    assert (v.usage == ANNOTATED).all()
    v[1, 2:4].set_usage(OBSERVED)
    assert (v.usage == OBSERVED).sum() == 2


def test_clamp():
    t = torch.ones(3, 4)
    v = Var(t, Range[4])
    v2 = v[2, :]
    with pytest.raises(ValueError):
        v2.set_usage(OBSERVED)
    v.set_usage(OBSERVED)
    # nothing annotated, so nothing clamped
    v.clamp_annotated()
    assert (v2.usage == OBSERVED).all()

    # mark as annotated
    v2.set_usage(ANNOTATED)
    v.clamp_annotated()
    assert (v2.usage == CLAMPED).all()
    assert (v.usage == CLAMPED).sum() == 4

    # unclamp
    v2.unclamp_annotated()
    assert (v2.usage == ANNOTATED).all()
    assert (v.usage == ANNOTATED).sum() == 4


def test_compose_single():
    assert compose_single(slice(3, 14, 2), slice(2, 6, 3), 100) == slice(7, 15, 6)


def test_compose():
    shape = (4, 10, 12, 7)
    first = (slice(None), 3, slice(3, 9))
    second = (2, slice(1, 3), slice(5))
    expected_combined = (2, 3, slice(4, 6, 1), slice(0, 5, 1))

    assert compose(shape, first, second) == expected_combined

    other_first = (slice(2, 4), slice(None), slice(2, 7))
    other_second = (0, 3, slice(2, -1), slice(5))

    assert compose(shape, other_first, other_second) == expected_combined


def test_nested():
    t = torch.zeros(4, 10, 12, 7)
    v = Var(t, Range[4])
    va = v[:, 3, 3:9][2, 1:3, :5]
    va.set_tensor(1)
    assert va.shape == (2, 5)
    assert va.tensor.sum() == 10

    vb = v[2:4, :, 2:7][0, 3, 2:-1, :5]
    assert vb.shape == (2, 5)
    assert vb.tensor.sum() == 10

    assert v.tensor.sum() == 10
    assert va == vb

    v.set_tensor(3.)
    assert va.tensor.sum() == 30
    assert vb.tensor.sum() == 30


def test_domain():
    t = torch.ones(3, 4)
    v = Var(t, Range[4])
    v2 = v[2:, :]
    assert v2.domain == v.domain


def test_eq():
    t = torch.ones(3, 4)
    t2 = torch.ones(3, 4)
    v1 = Var(t, Range[4])
    v2 = Var(t, Range[4])
    v3 = v1[2:, :]
    v4 = v2[2:, :]
    assert v1 == v2
    assert v3 == v4
    assert v1 != v3
    v5 = Var(t2, Range[4])
    assert v1 != v5


def test_dict():
    t = torch.ones(3, 4)
    t2 = torch.ones(3, 4)
    v1 = Var(t, Range[4])
    v2 = Var(t, Range[4])
    assert hash(v1) == hash(v2)
    v3 = v1[2:, :]
    v4 = v2[2:, :]
    assert hash(v3) == hash(v4)
    v5 = Var(t2, Range[4])
    d: Dict[VarBase, Tuple[int, ...]] = dict()
    d[v1] = (1, 2)
    d[v3] = (3, 4)
    d[v5] = (5,)
    assert d[v1] == (1, 2)
    assert d[v2] == (1, 2)
    assert d[v3] == (3, 4)
    assert d[v4] == (3, 4)
    assert d[v5] == (5,)


def test_var_from_tensor_usage():
    t = torch.ones(3, 4)
    v = Var(Range[4], t)
    assert v.tensor is t
    assert list(v.domain) == [0, 1, 2, 3]
    v.set_usage(DEFAULT)
    assert v.usage.shape == t.shape
    assert (v.usage == DEFAULT).all()


def test_var_from_tensor_usage_dom():
    t = torch.ones(3, 4)
    v = Var(t, LATENT, Range[4])
    assert v.tensor is t
    assert list(v.domain) == [0, 1, 2, 3]
    assert v.usage.shape == t.shape
    assert (v.usage == LATENT).all()


def test_var_from_usage_tensor_dom():
    t = torch.ones(3, 4)
    v = Var(LATENT, t, Range[4])
    assert v.tensor is t
    assert list(v.domain) == [0, 1, 2, 3]
    assert v.usage.shape == t.shape
    assert (v.usage == LATENT).all()


def test_var_from_usage_dom_tensor():
    t = torch.ones(3, 4)
    v = Var(LATENT, Range[4], t)
    assert v.tensor is t
    assert list(v.domain) == [0, 1, 2, 3]
    assert v.usage.shape == t.shape
    assert (v.usage == LATENT).all()


def test_pad_and_stack():
    vs = [
        Var(LATENT, Range[4], torch.ones(3, 4)),
        Var(LATENT, Range[4], torch.ones(2, 10)),
        Var(LATENT, Range[4], torch.ones(4, 7)),
    ]
    v = Var.pad_and_stack(vs)
    assert v.domain == Range[4]
    assert (v.usage == LATENT).sum() == (3 * 4 + 2 * 10 + 4 * 7)
    assert v.shape == (3, 4, 10)


def test_var_field():
    v = VarField()
    with pytest.raises(NotImplementedError):
        v.tensor
    with pytest.raises(NotImplementedError):
        v.set_tensor(1.0)
    with pytest.raises(NotImplementedError):
        v.usage
    with pytest.raises(NotImplementedError):
        v.set_usage(DEFAULT)
    with pytest.raises(NotImplementedError):
        v.domain
    with pytest.raises(NotImplementedError):
        v.original_tensor
    with pytest.raises(NotImplementedError):
        v.ndslice
