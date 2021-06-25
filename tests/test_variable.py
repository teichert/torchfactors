from typing import Dict, Tuple

import pytest
import torch
from torchfactors import (ANNOTATED, CLAMPED, DEFAULT, LATENT, OBSERVED,
                          Var, VarUsage)
from torchfactors.domain import Range
from torchfactors.variable import TensorVar, VarField


def test_usage1():
    assert VarUsage.DEFAULT == VarUsage.OBSERVED


def test_variable():
    t = torch.ones(3, 4)
    v = TensorVar(t, Range(4))
    assert len(v.domain) == 4
    assert v.tensor is t
    assert v.shape == (3, 4)
    v.set_usage(OBSERVED)
    assert isinstance(v.usage, torch.Tensor)
    print(v.usage == VarUsage.OBSERVED)
    assert (v.usage == VarUsage.OBSERVED).all()
    assert v.ndslice == (...,)
    assert v.original_tensor is t
    assert v.origin is v


def test_shape():
    t = torch.ones(3, 4)
    v = TensorVar(t, Range(4))
    v2 = v[1:3, 2:4]
    assert v2.origin is v
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
    v = TensorVar(t, Range(4))
    v.set_usage(ANNOTATED)
    assert (v.usage == ANNOTATED).all()
    v[1, 2:4].set_usage(OBSERVED)
    assert (v.usage == OBSERVED).sum() == 2


def test_clamp():
    t = torch.ones(3, 4)
    v = TensorVar(t, Range(4))
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


def test_nested():
    t = torch.zeros(4, 10, 12, 7)
    v = TensorVar(t, Range(4))
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
    v = TensorVar(t, Range(4))
    v2 = v[2:, :]
    assert v2.domain == v.domain


def test_eq():
    t = torch.ones(3, 4)
    t2 = torch.ones(3, 4)
    v1 = TensorVar(t, Range(4))
    v2 = TensorVar(t, Range(4))
    v3 = v1[2:, :]
    v4 = v2[2:, :]
    assert v1 == v2
    assert v3 == v4
    assert v1 != v3
    v5 = TensorVar(t2, Range(4))
    assert v1 != v5


def test_dict():
    t = torch.ones(3, 4)
    t2 = torch.ones(3, 4)
    v1 = TensorVar(t, Range(4))
    v2 = TensorVar(t, Range(4))
    assert hash(v1) == hash(v2)
    v3 = v1[2:, :]
    v4 = v2[2:, :]
    assert hash(v3) == hash(v4)
    v5 = TensorVar(t2, Range(4))
    d: Dict[Var, Tuple[int, ...]] = dict()
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
    v = TensorVar(Range(4), t)
    assert v.tensor is t
    assert list(v.domain) == [0, 1, 2, 3]
    v.set_usage(DEFAULT)
    assert v.usage.shape == t.shape
    assert (v.usage == DEFAULT).all()


def test_var_from_tensor_usage_dom():
    t = torch.ones(3, 4)
    v = TensorVar(t, LATENT, Range(4))
    assert v.tensor is t
    assert list(v.domain) == [0, 1, 2, 3]
    assert v.usage.shape == t.shape
    assert (v.usage == LATENT).all()


def test_var_from_usage_tensor_dom():
    t = torch.ones(3, 4)
    v = TensorVar(LATENT, t, Range(4))
    assert v.tensor is t
    assert list(v.domain) == [0, 1, 2, 3]
    assert v.usage.shape == t.shape
    assert (v.usage == LATENT).all()


def test_var_from_usage_dom_tensor():
    t = torch.ones(3, 4)
    v = TensorVar(LATENT, Range(4), t)
    assert v.tensor is t
    assert list(v.domain) == [0, 1, 2, 3]
    assert v.usage.shape == t.shape
    assert (v.usage == LATENT).all()


def test_pad_and_stack():
    vs = [
        TensorVar(LATENT, Range(4), torch.ones(3, 4)),
        TensorVar(LATENT, Range(4), torch.ones(2, 10)),
        TensorVar(LATENT, Range(4), torch.ones(4, 7)),
    ]
    v = TensorVar.pad_and_stack(vs)
    assert v.domain == Range(4)
    assert (v.usage == LATENT).sum() == (3 * 4 + 2 * 10 + 4 * 7)
    assert v.shape == (3, 4, 10)

    a, b, c = v.unstack()
    assert a.domain == Range(4)
    assert (a.usage == LATENT).sum() == 3 * 4
    assert a.shape == (3, 4)

    assert b.domain == Range(4)
    assert (b.usage == LATENT).sum() == 2 * 10
    assert b.shape == (2, 10)

    assert c.domain == Range(4)
    assert (c.usage == LATENT).sum() == 4 * 7
    assert c.shape == (4, 7)

    with pytest.raises(ValueError):
        a.unstack()


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
    with pytest.raises(NotImplementedError):
        v.origin
    with pytest.raises(NotImplementedError):
        v[3]


def test_grad_through_stack():
    vs = [
        TensorVar(LATENT, Range(4), torch.ones(3, 4, requires_grad=True)),
        TensorVar(LATENT, Range(4), torch.ones(2, 10, requires_grad=True)),
        TensorVar(LATENT, Range(4), torch.ones(4, 7, requires_grad=True)),
    ]
    v = TensorVar.pad_and_stack(vs)
    (v.tensor[v.usage == LATENT].sum() * 2).backward()
    assert (vs[0].tensor.grad == 2).all()
    assert (vs[1].tensor.grad == 2).all()
    assert (vs[2].tensor.grad == 2).all()


def test_grad_through_and_unstack():
    vs = [
        TensorVar(LATENT, Range(4), torch.ones(3, 4, requires_grad=True)),
        TensorVar(LATENT, Range(4), torch.ones(2, 10, requires_grad=True)),
        TensorVar(LATENT, Range(4), torch.ones(4, 7, requires_grad=True)),
    ]
    v = TensorVar.pad_and_stack(vs)
    v_outs = v.unstack()
    for i, vout in enumerate(v_outs):
        vout.tensor.prod().backward(retain_graph=True)

    assert (vs[0].tensor.grad == 1).all()
    assert (vs[1].tensor.grad == 1).all()
    assert (vs[2].tensor.grad == 1).all()


# # This one doesn't work
# def test_grad_through_and_unstack2():
#     vs = [
#         TensorVar(LATENT, Range(4), torch.ones(3, 4, requires_grad=True)),
#         TensorVar(LATENT, Range(4), torch.ones(2, 10, requires_grad=True)),
#         TensorVar(LATENT, Range(4), torch.ones(4, 7, requires_grad=True)),
#     ]
#     v = TensorVar.pad_and_stack(vs)
#     v_outs = v.unstack()
#     for i, vout in enumerate(v_outs):
#         vout.tensor.prod().backward()
#         assert (vs[i].tensor.grad == 1).all()


def test_var_field_order():
    v = VarField(OBSERVED, Range(10))
    assert v._usage == OBSERVED
    assert v._domain == Range(10)

    v = VarField(Range(10), OBSERVED)
    assert v._usage == OBSERVED
    assert v._domain == Range(10)


# def test_as_used():
#     v = TensorVar(Range(5),
#                   tensor=torch.tensor([
#                       3,   # 0
#                       2,   # 1
#                       0,   # 0
#                       1,   # 1
#                       1,   # 2
#                       2,   # 3
#                       0,   # 4
#                       3,   # 5
#                       2,   # 6
#                       0]),  # 7
#                   usage=torch.tensor([
#                       OBSERVED, OBSERVED,
#                       ANNOTATED, ANNOTATED,
#                       CLAMPED, CLAMPED,
#                       PADDING, PADDING,
#                       LATENT, LATENT,
#                   ]))
#     assert (v.usage_mask == torch.tensor([
#         [0, 0, 0, 1, 0],
#         [0, 0, 1, 0, 0],
#         [1, 1, 1, 1, 1],
#         [1, 1, 1, 1, 1],
#         [0, 1, 0, 0, 0],
#         [0, 0, 1, 0, 0],
#         [float('nan'), 0, 0, 0, 0],
#         [0, 0, 0, float('nan'), 0],
#         [1, 1, 1, 1, 1]
#     ]).log()).all()
