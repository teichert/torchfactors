from typing import Any, Dict, Tuple, cast

import pytest
import torch
import torchfactors
import torchfactors as tx
from torchfactors import (ANNOTATED, CLAMPED, DEFAULT, LATENT, OBSERVED,
                          PADDING, Var, VarUsage)
from torchfactors.domain import Range
from torchfactors.factor import Factor
from torchfactors.subject import Environment
from torchfactors.variable import TensorVar, VarField, at

# def test_at():
#     t = torch.tensor([
#         [
#             [1, 2, 3, 4],
#             [5, 6, 7, 8],
#         ],
#         [
#             [11, 12, 13, 14],
#             [15, 16, 17, 18],
#         ],
#         [
#             [21, 22, 23, 24],
#             [25, 26, 27, 28],
#         ],
#     ])
#     s = [slice(None), tx.gslice(1), tx.gslice(2, 0, 3)]
#     expected = torch.tensor([
#         [
#             [7, 5, 8],
#         ],
#         [
#             [17, 15, 18],
#         ],
#         [
#             [27, 25, 28],
#         ],
#     ])
#     out = at(t, s)
#     assert out.tolist() == expected.tolist()


# def test_at2():
#     t = torch.tensor([
#         [
#             [1, 2, 3, 4],
#             [5, 6, 7, 8],
#         ],
#         [
#             [11, 12, 13, 14],
#             [15, 16, 17, 18],
#         ],
#         [
#             [21, 22, 23, 24],
#             [25, 26, 27, 28],
#         ],
#     ])
#     s = [tx.gslice(2, 0), 1, tx.gslice(2, 0, 3)]
#     expected = torch.tensor([
#         [27, 25, 28],
#         [7, 5, 8],
#     ])
#     out = at(t, s)
#     assert out.tolist() == expected.tolist()
#
#
# def test_at4():
#     t = torch.tensor([
#         [
#             [1, 2, 3, 4],
#             [5, 6, 7, 8],
#         ],
#         [
#             [11, 12, 13, 14],
#             [15, 16, 17, 18],
#         ],
#         [
#             [21, 22, 23, 24],
#             [25, 26, 27, 28],
#         ],
#     ])
#     s = [tx.gslice(2, 0), 1, tx.gslice(2, 0, 3)]
#     out = at(t, -1)
#     expected = torch.tensor(
#         [
#             [21, 22, 23, 24],
#             [25, 26, 27, 28],
#         ],
#     )
#     assert out.tolist() == expected.tolist()


def test_at3():
    t = torch.tensor([
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ],
        [
            [11, 12, 13, 14],
            [15, 16, 17, 18],
        ],
        [
            [21, 22, 23, 24],
            [25, 26, 27, 28],
        ],
    ])
    out = at(t, slice(1, 3))
    expected = torch.tensor([
        [
            [11, 12, 13, 14],
            [15, 16, 17, 18],
        ],
        [
            [21, 22, 23, 24],
            [25, 26, 27, 28],
        ],
    ])
    assert out.tolist() == expected.tolist()


def test_at_jagged():
    t = torch.tensor([
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ],
        [
            [11, 12, 13, 14],
            [15, 16, 17, 18],
        ],
        [
            [21, 22, 23, 24],
            [25, 26, 27, 28],
        ],
    ])
    s = (slice(None), 1, tx.gdrop(1, 0, 3))
    out = at(t, s)
    expected = torch.tensor([6, 15, 28])
    assert out.tolist() == expected.tolist()


def test_at_jagged2():
    t = torch.tensor([
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ],
        [
            [11, 12, 13, 14],
            [15, 16, 17, 18],
        ],
        [
            [21, 22, 23, 24],
            [25, 26, 27, 28],
        ],
    ])
    expected = torch.tensor([
        7,
        13,
        27,
    ])
    s = (slice(None), tx.gdrop(1, 0, 1), 2)
    out = at(t, s)
    assert out.tolist() == expected.tolist()


def test_usage1():
    assert VarUsage.DEFAULT == VarUsage.LATENT


def test_set_tensor_with_salar_for_first_time():
    v = TensorVar(Range(4))
    v.set_tensor(5)
    assert v.shape == ()


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
    assert v.origin.tensor is t
    assert v.origin is v


def test_vtensor():
    v = torchfactors.vtensor(torch.ones(3, 4).tolist())
    v._domain = Range(4)
    assert len(v.domain) == 4
    assert v.shape == (3, 4)
    v.set_usage(OBSERVED)
    assert isinstance(v.usage, torch.Tensor)
    print(v.usage == VarUsage.OBSERVED)
    assert (v.usage == VarUsage.OBSERVED).all()
    assert v.ndslice == (...,)
    assert v.origin.tensor is v.tensor
    assert v.origin is v


def test_shape():
    t = torch.ones(3, 4)
    v = TensorVar(t, Range(4))
    v2 = v[1:3, 2:4]
    assert v2.origin is v
    assert v2.origin.tensor is t
    assert v2.ndslice == (slice(1, 3, 1), slice(2, 4, 1))
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


def test_bad_var_usage():
    v = TensorVar(domain=Range(10))
    with pytest.raises(TypeError):
        # need to have a usage before you can access it
        v.usage


def test_bad_var_usage2():
    v = TensorVar(domain=Range(10))
    with pytest.raises(ValueError):
        # need to have a tensor before subscripting
        v[2, :]


def test_bad_var_usage3():
    v = TensorVar(domain=Range(10))
    v.tensor = torch.ones(4, 5)
    # need to have a tensor before subscripting
    v[2, :].set_usage(ANNOTATED)
    assert (v.usage == ANNOTATED).sum() == 5


def test_clamp():
    t = torch.ones(3, 4)
    v = TensorVar(t, Range(4))
    v2 = v[2, :]
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
        v._set_tensor(1.0)
    with pytest.raises(NotImplementedError):
        v.tensor
    with pytest.raises(NotImplementedError):
        v.usage
    with pytest.raises(NotImplementedError):
        v._set_usage(DEFAULT)
    with pytest.raises(NotImplementedError):
        v.domain
    with pytest.raises(NotImplementedError):
        v.ndslice
    with pytest.raises(NotImplementedError):
        v.origin
    with pytest.raises(NotImplementedError):
        v[3]


# test_var_field()


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


def test_grad_through_and_unstack2():
    vs = [
        TensorVar(LATENT, Range(4), torch.ones(3, 4, requires_grad=True)),
        TensorVar(LATENT, Range(4), torch.ones(2, 10, requires_grad=True)),
        TensorVar(LATENT, Range(4), torch.ones(4, 7, requires_grad=True)),
    ]
    v = TensorVar.pad_and_stack(vs)
    v_outs = v.unstack()
    for i, vout in enumerate(v_outs):
        vout.tensor.prod().backward(retain_graph=True)
        assert (vs[i].tensor.grad == 1).all()


def test_var_field_order():
    v = VarField(OBSERVED, Range(10))
    assert v._usage == OBSERVED
    assert v._domain == Range(10)

    v = VarField(Range(10), OBSERVED)
    assert v._usage == OBSERVED
    assert v._domain == Range(10)


def test_as_used():
    v = TensorVar(Range(5),
                  tensor=torch.tensor([
                      3,   # 0
                      2,   # 1
                      0,   # 0
                      1,   # 1
                      1,   # 2
                      2,   # 3
                      0,   # 4
                      3,   # 5
                      2,   # 6
                      0]),  # 7
                  usage=torch.tensor([
                      OBSERVED, OBSERVED,
                      ANNOTATED, ANNOTATED,
                      CLAMPED, CLAMPED,
                      PADDING, PADDING,
                      LATENT, LATENT,
                  ]))
    expected_possible = torch.tensor([
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]
    ]).bool()
    assert (v.is_possible == expected_possible).all()
    expected_padding = torch.tensor([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]).bool()
    assert (v.is_padding == expected_padding).all()


def test_broken_usage():
    @tx.dataclass
    class BitLanguageSubject(tx.Subject):
        bits: tx.Var = tx.VarField(tx.Range(2), tx.ANNOTATED)  # TensorType[..., len, 8]

        @classmethod
        def from_string(cls, x: str) -> 'BitLanguageSubject':
            return BitLanguageSubject(
                bits=tx.vtensor(
                    [
                        [bit == '1' for bit in format(
                            ord(ch) % (2 ** 8),
                            '08b')]
                        for ch in x]))
    data = BitLanguageSubject.from_string('t')
    assert (data.bits.usage == tx.ANNOTATED).all()
    data = data.clamp_annotated()
    assert (data.bits.usage == tx.CLAMPED).all()
    data = data.unclamp_annotated()
    assert (data.bits.usage == tx.ANNOTATED).all()


def test_flex_domain():
    domain = tx.FlexDomain('property')
    m = tx.Model[tx.Subject]()
    input = ['this', 'is', 'this', 'test', 'is', 'this']
    ids = m.domain_ids(domain, input)
    values = m.domain_values(domain, ids)
    assert len(set(ids.tolist())) == 3
    assert len(domain) == 4
    assert set(domain) == {domain.unk, 'this', 'is', 'test'}
    assert values == input


def test_frozen_flex_domain():
    domain = tx.FlexDomain('property')
    m = tx.Model[tx.Subject]()
    m.domain_ids(domain, ['this', 'is', 'this', 'test', 'is', 'this'])
    domain.freeze()
    ids = m.domain_ids(domain, ['now', 'what', 'is', 'this', 'test', '?'])
    values = m.domain_values(domain, ids)
    assert values == [
        domain.unk,
        domain.unk,
        'is',
        'this',
        'test',
        domain.unk]
    assert len(domain) == 4
    assert domain.get_value(100) == domain.unk


def test_flatten():
    v = tx.TensorVar(torch.tensor([[4, 5, 1], [4, 2, 3]]), Range(6), tx.ANNOTATED)
    assert v.flatten().tolist() == [4, 5, 1, 4, 2, 3]
    v.usage = torch.tensor([
        [tx.ANNOTATED, tx.LATENT, tx.LATENT],
        [tx.LATENT, tx.LATENT, tx.ANNOTATED]])
    assert v.flatten(usage=tx.ANNOTATED).tolist() == [4, 3]
    assert v.flatten(usage=tx.LATENT).tolist() == [5, 1, 4, 2]


def test_no_late_change():
    v = tx.TensorVar(torch.tensor([[4, 5, 1], [4, 2, 3]]), Range(6), tx.ANNOTATED)
    v2 = v[1]
    v.is_padding
    with pytest.raises(TypeError):
        v.usage[1] = VarUsage.CLAMPED
    with pytest.raises(TypeError):
        v2.tensor[1] = 2


def test_no_late_change2():
    v = tx.TensorVar(torch.tensor([[4, 5, 1], [4, 2, 3]]), Range(6), tx.ANNOTATED)
    v2 = v[1]
    v2.is_possible
    with pytest.raises(TypeError):
        # can't change tensor after getting is_possible
        v.tensor[1] = 2
    with pytest.raises(TypeError):
        # can't change usage after getting is_possible
        v2.usage[1] = VarUsage.CLAMPED


def test_clone():
    v = tx.TensorVar(torch.tensor([[4, 5, 1], [4, 2, 3]]), Range(6), tx.ANNOTATED)
    v2 = v.clone()
    v.usage[(...,)] = tx.LATENT
    assert cast(torch.Tensor, (v.usage == tx.LATENT)).all()
    assert cast(torch.Tensor, (v2.usage == tx.ANNOTATED)).all()
    v2.tensor[(...,)] = 0
    assert v2.tensor.sum() == 0
    assert v.tensor.sum() > 0
    # make sure I can still clamp
    v.is_possible
    with pytest.raises(TypeError):
        # can't change usage after getting is_possible
        v.clamp_annotated()
    with pytest.raises(TypeError):
        # can't change tensor after getting is_possible
        v.set_tensor(3.0)
    v2.clamp_annotated()
    assert cast(torch.Tensor, (v2.usage == tx.CLAMPED)).all()


def test_environment_vars():
    env = Environment()
    v: Var = VarField()
    v2: Var = VarField()
    assert env.variable('a', lambda: v) is v
    assert env.variable('a', lambda: v2) is v
    assert env.variable('a', lambda: v2) is not v2


def test_environment_factors():
    env = Environment()
    model = tx.Model[Any]()
    v: Var = tx.vtensor([3, 4, 5])
    f: Factor = tx.LinearFactor(model.namespace('a'), v)
    f2: Factor = tx.LinearFactor(model.namespace('b'), v)
    assert env.factor('b', lambda: f) is f
    assert env.factor('b', lambda: f2) is f
    assert env.factor('b', lambda: f2) is not f2


def test_list_variable():
    v = tx.vtensor([
        [1, 2, 3],
        [4, 5, 6],
    ])
    v1 = v[..., 0]
    v2 = v[:, tx.gdrop(0, 0)]
    a = v1.tensor.tolist()
    b = v2.tensor.tolist()
    assert a == b


def test_list_variable2():
    v = tx.vtensor([
        [1, 2, 3],
        [4, 5, 6],
    ])
    v1 = v[..., 0]
    v2 = v[:, tx.gdrop(0)]
    a = v1.tensor.tolist()
    b = v2.tensor.tolist()
    assert a == b
