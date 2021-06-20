import torch
from torchfactors import ANNOTATED, CLAMPED, OBSERVED, VarUsage
from torchfactors.domain import Range
from torchfactors.variable import Var, compose, compose_single


def test_usage1():
    assert VarUsage.DEFAULT == VarUsage.OBSERVED


def test_variable():
    t = torch.ones(3, 4)
    v = Var(t, Range[4])
    assert len(v.domain) == 4
    assert v.tensor is t
    assert v.shape == (3, 4)
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
    # nothing annotated, so nothing clamped
    v2.set_usage(OBSERVED)
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
    v3 = v1[2:, :]
    v4 = v2[2:, :]
    v5 = Var(t2, Range[4])
    d = dict()
    d[v1] = (1, 2)
    d[v3] = (3, 4)
    d[v5] = (5,)
    assert d[v1] == (1, 2)
    assert d[v2] == (1, 2)
    assert d[v3] == (3, 4)
    assert d[v4] == (3, 4)
    assert d[v5] == (5,)
