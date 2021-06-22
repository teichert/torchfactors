from dataclasses import dataclass
from typing import List

import pytest
import torch
from torchfactors import LATENT, OBSERVED, Range, Subject, TensorVar, VarField
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


#     assert u.observations.domain == Range[10]
#     assert u.hidden.domain == Range[4]
#     assert u.observations.shape == (6,)
#     assert u.hidden.shape == (6,)
#     assert (u.observations.usage == OBSERVED).all()
#     assert (u.hidden.usage == LATENT).all()
