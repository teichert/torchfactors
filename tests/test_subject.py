from dataclasses import dataclass
from typing import ClassVar, List

import pytest
import torch
import torchfactors as tx
from torch.utils.data import Dataset
from torchfactors import (ANNOTATED, CLAMPED, LATENT, OBSERVED, Range, Subject,
                          TensorFactor, TensorVar, VarField)
from torchfactors.subject import ChainDataset, Environment, ListDataset
from torchfactors.variable import Var, vtensor


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
        observations: Var = VarField(Range(10), OBSERVED)
        hidden: Var = VarField(Range(4), LATENT, shape=observations)

    v = TensorVar(torch.tensor([2, 0, 1, 2, 3, 8]))
    u = Utterance(v)
    assert u.observations.domain == Range(10)
    assert u.hidden.domain == Range(4)
    assert u.observations.shape == (6,)
    assert u.hidden.shape == (6,)
    assert (u.observations.usage == OBSERVED).all()
    assert (u.hidden.usage == LATENT).all()


def test_implicit():
    @dataclass
    class Utterance(Subject):
        observations: Var = VarField(Range(10), OBSERVED)
        other: Var = VarField(Range(4), OBSERVED, shape=observations)

    with pytest.raises(ValueError):
        Utterance(TensorVar(torch.tensor([1, 3, 2, 4, 3, 5, 4]))),


def test_only_implicit():
    @dataclass
    class Utterance(Subject):
        v: Var = VarField(Range(4), LATENT, shape=(3, 4), init=torch.ones)

    u = Utterance()
    assert (u.v.tensor == 1).sum() == 3 * 4
    assert u.v.shape == (3, 4)


def test_no_shape():
    with pytest.raises(ValueError):
        @dataclass
        class Utterance(Subject):
            v: Var = VarField(Range(4), LATENT)

        Utterance()


def test_stack_zero():
    with pytest.raises(ValueError):
        @dataclass
        class Utterance(Subject):
            v: Var = VarField(Range(4), LATENT)

        Utterance.stack([])


def test_stack_twice():
    @dataclass
    class Utterance(Subject):
        v: Var = VarField(Range(4), LATENT, shape=(3, 4))

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
        items1: Var = VarField(Range(4), ANNOTATED)
        items2: Var = VarField(Range(4), ANNOTATED)

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
        observations: Var = VarField(Range(10), OBSERVED)
        hidden: Var = VarField(Range(4), LATENT, shape=observations)
    u1 = Utterance(1, 6, TensorVar(torch.tensor([1, 3, 2, 4, 3, 5, 4])))
    assert u1.list('id1') == [1]
    assert u1.list('id2') == [6]

    data = [
        u1,
        Utterance(2, 7, TensorVar(torch.tensor([2, 4, 3, 5]))),
        Utterance(3, 8, TensorVar(torch.tensor([4, 6, 5]))),
        Utterance(4, 9, TensorVar(torch.tensor([3, 2, 4, 3, 5, 4, 6, 5]))),
        Utterance(5, 0, TensorVar(torch.tensor([3]))),
    ]
    loader = Utterance.data_loader(data, batch_size=2)
    instances: List[Utterance] = list(loader)

    class MyDataset(Dataset):
        def __len__(self):
            return len(data)

        def __getitem__(self, idx: int):
            return data[idx]

    loader2 = Utterance.data_loader(MyDataset(), batch_size=2)
    instances2 = list(loader2)
    assert [x.id1 for x in instances] == [x.id1 for x in instances2]
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
    assert len(u1) == 1
    assert len(a) == 2
    assert len(b) == 2
    assert len(c) == 1
    assert len(x1) == 1
    assert len(x2) == 1
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


def test_full_batch():
    @dataclass
    class Utterance(Subject):
        id1: int
        id2: int
        observations: Var = VarField(Range(10), OBSERVED)
        hidden: Var = VarField(Range(4), LATENT, shape=observations)

    u1 = Utterance(1, 6, TensorVar(torch.tensor([1, 3, 2, 4, 3, 5, 4])))

    data = [
        u1,
        Utterance(2, 7, TensorVar(torch.tensor([2, 4, 3, 5]))),
        Utterance(3, 8, TensorVar(torch.tensor([4, 6, 5]))),
        Utterance(4, 9, TensorVar(torch.tensor([3, 2, 4, 3, 5, 4, 6, 5]))),
        Utterance(5, 0, TensorVar(torch.tensor([3]))),
    ]
    loader = Utterance.data_loader(data, batch_size=-1)
    instances: List[Utterance] = list(loader)
    assert len(instances) == 1
    assert instances[0].observations.shape == (5, 8)


def test_variables():
    @dataclass
    class Utterance(Subject):
        id1: int
        id2: int
        observations: Var = VarField(Range(10), OBSERVED)
        hidden: Var = VarField(Range(4), LATENT, shape=observations)

    class Middle(Utterance):
        test: int = 10

    @dataclass
    class ExtendedUtterance(Middle):
        other1: Var = VarField()
        other2: Var = VarField()

    u = ExtendedUtterance(
        3, 4,
        TensorVar(torch.ones(3)),
        other1=TensorVar(torch.ones(9)),
        other2=TensorVar(torch.ones(10)))

    assert len(u.variables) == 4
    assert u.variables[0].domain == Range(10)
    assert u.variables[1].domain == Range(4)
    assert u.variables[2].tensor.shape == (9,)
    assert u.variables[3].tensor.shape == (10,)


def test_var_check():
    @dataclass
    class MySubject(Subject):
        v: Var = VarField()

    with pytest.raises(TypeError):
        # passing in tensor rather than var
        MySubject(torch.tensor(3.))  # type: ignore


def test_bad_env1():
    env = Environment()
    with pytest.raises(KeyError):
        env.variable('test')


def test_good_env1():
    env = Environment()
    v = env.variable('test', lambda: vtensor([2, 9, 3]))
    assert env.variable('test') is v


def test_bad_env2():
    env = Environment()
    with pytest.raises(KeyError):
        env.factor('test')


def test_good_env2():
    env = Environment()
    v = tx.TensorVar(torch.tensor([1, 2, 3]), tx.Range(5))
    f = env.factor('test', lambda: TensorFactor(v, tensor=torch.ones(3, 5)))
    assert env.factor('test') is f


def test_subject_shape1():
    @tx.dataclass
    class MySubject(tx.Subject):
        a: Var = VarField(Range(5))
        b: Var = VarField(Range(5), shape=a, usage=LATENT)

    subject = MySubject(vtensor([[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7]]))
    assert subject.b.shape == (2, 7)


def test_subject_shape2():
    @tx.dataclass
    class MySubject(tx.Subject):
        a: Var = VarField(Range(5), shape=(3, 5), usage=LATENT)

    subject = MySubject()
    assert subject.a.shape == (3, 5)


def test_subject_bad_shape1():
    @tx.dataclass
    class MySubject(tx.Subject):
        a: Var = VarField(Range(5))
        b: Var = VarField(Range(5), shape=a)

    with pytest.raises(ValueError):
        MySubject(vtensor([[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7]]))


def test_subject_bad_shape1b():
    @tx.dataclass
    class MySubject(tx.Subject):
        a: Var = VarField(Range(5))
        b: Var = VarField(Range(5), shape=a, usage=LATENT)

    MySubject(vtensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]))

    with pytest.raises(ValueError):
        MySubject(vtensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]),
                  vtensor([[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]]))


def test_subject_bad_shape2():
    @tx.dataclass
    class MySubject(tx.Subject):
        a: Var = VarField(Range(5), shape=(3, 5))

    with pytest.raises(ValueError):
        # cannot specify tensor if already specified different shape
        MySubject(vtensor([1, 2, 3]))


def test_subject_matching_shape():
    @tx.dataclass
    class MySubject(tx.Subject):
        a: Var = VarField(Range(5), shape=(3,))

    out = MySubject(vtensor([1, 2, 3]))
    assert out.a.tensor.shape == (3,)


def test_subject_clone():
    @tx.dataclass
    class MySubject(tx.Subject):
        a: Var = VarField(Range(5))
        b: Var = VarField(Range(5), shape=a, usage=LATENT)

    subject = MySubject(vtensor([[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7]]))

    subject2 = subject.clone()
    assert (subject2.a.tensor.tolist() ==
            [[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7]])
    subject2.a.tensor[(...,)] = 9
    assert (subject2.a.tensor == 9).all()
    assert (subject.a.tensor != 9).all()


def test_subject_clone2():
    @tx.dataclass
    class MySubject(tx.Subject):
        a: Var = VarField(Range(5))
        b: Var = VarField(Range(5), shape=a, usage=LATENT)

    subject = MySubject(vtensor([[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7]]))

    with pytest.raises(TypeError):
        # need to give actual device, not just name
        subject.clone(device='cpu')

    subject2 = subject.clone(torch.device('cpu'))
    assert (subject2.a.tensor.tolist() ==
            [[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7]])
    subject2.a.tensor[(...,)] = 9
    assert (subject2.a.tensor == 9).all()
    assert (subject.a.tensor != 9).all()
    assert subject2.a.tensor.device == torch.device('cpu')


def test_subject_clone3():
    @tx.dataclass
    class MySubject(tx.Subject):
        a: Var = VarField(Range(5))
        b: Var = VarField(Range(5), shape=a, usage=LATENT)

    subject = MySubject(vtensor([[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7]]))
    subject2 = subject.clone(torch.device('meta'))
    assert subject2.a.tensor.device == torch.device('meta')


def test_subject_clone4():
    @tx.dataclass
    class MySubject(tx.Subject):
        a: Var = VarField(Range(5))
        b: Var = VarField(Range(5), shape=a, usage=LATENT)

    subject = MySubject(vtensor([[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7]]))

    with pytest.raises(TypeError):
        # need to give actual device, not just name
        subject.to_device(device='cpu')  # type: ignore

    subject2 = subject.to_device(torch.device('cpu'))
    assert (subject2.a.tensor.tolist() ==
            [[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7]])
    subject2.a.tensor[(...,)] = 9
    assert (subject2.a.tensor == 9).all()
    assert (subject.a.tensor == 9).all()
    assert subject2.a.tensor.device == torch.device('cpu')


def test_subject_clone5():
    @tx.dataclass
    class MySubject(tx.Subject):
        a: Var = VarField(Range(5))
        b: Var = VarField(Range(5), shape=a, usage=LATENT)

    subject = MySubject(vtensor([[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7]]))
    subject2 = subject.to_device(torch.device('meta'))
    assert subject2.a.tensor.device == torch.device('meta')


def test_subject_ndims():
    @tx.dataclass
    class MySubject(tx.Subject):
        a: Var = VarField(Range(5))
        b: Var = VarField(Range(5), shape=a, usage=LATENT)

    with pytest.raises(NotImplementedError):
        MySubject.a.ndims

    subject = MySubject(vtensor([[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7]]))
    assert subject.a.ndims == 2
    assert subject.b.ndims == 2


def test_subject_ndims2():
    @tx.dataclass
    class MySubject(tx.Subject):
        a: Var = VarField(Range(5))
        b: Var = VarField(Range(5), shape=a, usage=LATENT)

    subject = MySubject(
        tx.TensorVar(torch.tensor([[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7]]),
                     ndims=1))
    assert subject.a.ndims == 1
    assert subject.b.ndims == 1

    subject2 = MySubject(
        tx.TensorVar(torch.tensor([[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7]]),
                     ndims=1),
        tx.TensorVar(ndims=0))
    assert subject2.a.ndims == 1
    assert subject2.b.ndims == 0


# # Maybe sometime, it will be useful to do it this way
# def test_subject_ndims3():
#     @tx.dataclass
#     class MySubject(tx.Subject):
#         a: Var = VarField(Range(5))
#         b: Var = VarField(Range(5), shape=a, usage=LATENT, ndims=0)

#     subject = MySubject(
#         tx.TensorVar(torch.tensor([[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7]]),
#                      ndims=1))
#     assert subject.a.ndims == 1
#     assert subject.b.ndims == 0


def test_specify_optional():
    @tx.dataclass
    class MySubject(tx.Subject):
        a: Var = VarField(Range(5))
        b: Var = VarField(Range(5), shape=a, usage=OBSERVED)

    t = torch.tensor([[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7]])
    t2 = t + 1
    subject = MySubject(tx.TensorVar(t), tx.TensorVar(t2))
    assert subject.a.tensor.allclose(t)
    assert subject.b.tensor.allclose(t2)


def test_chain_dataset():
    @tx.dataclass
    class MySubject(tx.Subject):
        id: int
    dataset1 = ListDataset([MySubject(i) for i in range(5, 10)])
    dataset2 = ListDataset([MySubject(i) for i in range(15, 20)])
    dataset3 = ListDataset([MySubject(i - 3) for i in range(2, 4)])

    dataset4 = ChainDataset[MySubject]([dataset1, dataset2, dataset3])
    assert len(dataset4) == 12
    assert dataset4[0].id == 5
    assert dataset4[3].id == 8
    assert dataset4[7].id == 17
    assert dataset4[9].id == 19
    assert dataset4[10].id == -1
    assert dataset4[11].id == 0
    assert dataset4[-1].id == 0
    assert dataset4[-2].id == -1
    assert dataset4[-3].id == 19
    assert dataset4[-12].id == 5
    assert list(dataset4) == [
        MySubject(i) for i in [5, 6, 7, 8, 9, 15, 16, 17, 18, 19, -1, 0]
    ]

    with pytest.raises(IndexError):
        dataset4[12]

    with pytest.raises(IndexError):
        dataset4[-13]
