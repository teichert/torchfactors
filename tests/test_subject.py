from dataclasses import dataclass
from typing import ClassVar, List

import pytest
import torch
from torchfactors import (ANNOTATED, CLAMPED, LATENT, OBSERVED, Range, Subject,
                          TensorVar, VarField)
from torchfactors.variable import Var


def test_subject_nodataclass():
    with pytest.raises(ValueError):
        class MySubject(Subject):
            pass

        MySubject()


def test_subject_good():
    @dataclass
    class MySubject(Subject):
        pass

    MySubject()


def test_basic():
    @dataclass
    class Utterance(Subject):
        observations: Var = VarField(Range[10], OBSERVED)
        hidden: Var = VarField(Range[4], LATENT, shape=observations)

    v = TensorVar(torch.tensor([2, 0, 1, 2, 3, 8]))
    u = Utterance(v)
    assert u.observations.domain == Range[10]
    assert u.hidden.domain == Range[4]
    assert u.observations.shape == (6,)
    assert u.hidden.shape == (6,)
    assert (u.observations.usage == OBSERVED).all()
    assert (u.hidden.usage == LATENT).all()


def test_implicit():
    with pytest.raises(ValueError):
        @dataclass
        class Utterance(Subject):
            observations: Var = VarField(Range[10], OBSERVED)
            other: Var = VarField(Range[4], OBSERVED, shape=observations)

        Utterance(TensorVar(torch.tensor([1, 3, 2, 4, 3, 5, 4]))),


def test_only_implicit():
    @dataclass
    class Utterance(Subject):
        v: Var = VarField(Range[4], LATENT, shape=(3, 4), init=torch.ones)

    u = Utterance()
    assert (u.v.tensor == 1).sum() == 3 * 4
    assert u.v.shape == (3, 4)


def test_no_shape():
    with pytest.raises(ValueError):
        @dataclass
        class Utterance(Subject):
            v: Var = VarField(Range[4], LATENT)

        Utterance()


def test_stack_zero():
    with pytest.raises(ValueError):
        @dataclass
        class Utterance(Subject):
            v: Var = VarField(Range[4], LATENT)

        Utterance.stack([])


def test_stack_twice():
    @dataclass
    class Utterance(Subject):
        v: Var = VarField(Range[4], LATENT, shape=(3, 4))

    u = Utterance.stack([Utterance()])
    with pytest.raises(ValueError):
        Utterance.stack([u])


def test_no_vars():
    @dataclass
    class Utterance(Subject):
        i: int

    data = [Utterance(3), Utterance(4), Utterance(5)]
    combined = Utterance.stack(data)
    a, b, c = combined.unstack()
    assert a.i == 3
    assert b.i == 4
    assert c.i == 5


def test_no_fields():
    @dataclass
    class Utterance(Subject):
        i: ClassVar[int] = 10

    data = [Utterance(), Utterance(), Utterance()]
    combined = Utterance.stack(data)
    with pytest.raises(ValueError):
        combined.unstack()


def test_clamp_annotated():
    @dataclass
    class Utterance(Subject):
        items1: Var = VarField(Range[4], ANNOTATED)
        items2: Var = VarField(Range[4], ANNOTATED)

    u = Utterance(TensorVar(torch.ones(10)), TensorVar(torch.ones(5)))
    assert (u.items1.usage == ANNOTATED).sum() == 10
    assert (u.items2.usage == ANNOTATED).sum() == 5
    assert (u.items1.usage == CLAMPED).sum() == 0
    assert (u.items2.usage == CLAMPED).sum() == 0
    u.clamp_annotated()
    assert (u.items1.usage == ANNOTATED).sum() == 0
    assert (u.items2.usage == ANNOTATED).sum() == 0
    assert (u.items1.usage == CLAMPED).sum() == 10
    assert (u.items2.usage == CLAMPED).sum() == 5
    u.unclamp_annotated()
    assert (u.items1.usage == ANNOTATED).sum() == 10
    assert (u.items2.usage == ANNOTATED).sum() == 5
    assert (u.items1.usage == CLAMPED).sum() == 0
    assert (u.items2.usage == CLAMPED).sum() == 0


def test_stacked():
    @dataclass
    class Utterance(Subject):
        id1: int
        id2: int
        observations: Var = VarField(Range[10], OBSERVED)
        hidden: Var = VarField(Range[4], LATENT, shape=observations)

    data = [
        Utterance(1, 6, TensorVar(torch.tensor([1, 3, 2, 4, 3, 5, 4]))),
        Utterance(2, 7, TensorVar(torch.tensor([2, 4, 3, 5]))),
        Utterance(3, 8, TensorVar(torch.tensor([4, 6, 5]))),
        Utterance(4, 9, TensorVar(torch.tensor([3, 2, 4, 3, 5, 4, 6, 5]))),
        Utterance(5, 0, TensorVar(torch.tensor([3]))),
    ]
    loader = Utterance.data_loader(data, batch_size=2)
    instances: List[Utterance] = list(loader)
    assert len(instances) == 3
    a, b, c = instances
    assert a.id1 == 1
    assert b.id1 == 3
    assert c.id1 == 5
    assert a.list('id1') == [1, 2]
    assert b.list('id1') == [3, 4]
    assert c.list('id1') == [5]

    assert a.id2 == 6
    assert b.id2 == 8
    assert c.id2 == 0
    assert a.list('id2') == [6, 7]
    assert b.list('id2') == [8, 9]
    assert c.list('id2') == [0]

    assert a.observations.shape == (2, 7)
    assert b.observations.shape == (2, 8)
    assert c.observations.shape == (1, 1)

    assert a.hidden.shape == (2, 7)
    assert b.hidden.shape == (2, 8)
    assert c.hidden.shape == (1, 1)

    x1, x2 = a.unstack()
    assert x1.id1 == 1
    assert x1.id2 == 6

    assert x2.id1 == 2
    assert x2.id2 == 7

    assert x1.observations.shape == (7,)
    assert x2.observations.shape == (4,)
    assert x1.hidden.shape == (7,)
    assert x2.hidden.shape == (4,)

    assert x1.observations.tensor.shape == (7,)
    assert x2.observations.tensor.shape == (4,)

    assert x1.observations.usage.shape == (7,)
    assert x2.observations.usage.shape == (4,)


#     assert u.observations.domain == Range[10]
#     assert u.hidden.domain == Range[4]
#     assert u.observations.shape == (6,)
#     assert u.hidden.shape == (6,)
#     assert (u.observations.usage == OBSERVED).all()
#     assert (u.hidden.usage == LATENT).all()
