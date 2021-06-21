from dataclasses import dataclass

import pytest
from torchfactors import Subject


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


# def test_basic():
#     @dataclass
#     class Utterance(Subject):
#         observations: Var = VarField(Range[10], OBSERVED)
#         hidden: Var = VarField(Range[4], LATENT, shape=observations)

#     u = Utterance(Var(torch.tensor([2, 0, 1, 2, 3, 8])))
#     assert u.observations.domain == Range[10]
#     assert u.hidden.domain == Range[4]
#     assert u.observations.shape == (6,)
#     assert u.hidden.shape == (6,)
