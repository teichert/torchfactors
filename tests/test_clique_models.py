
import pytest
import torch
import torchfactors as tx
from torchfactors.clique import (BinaryScoresModule,
                                 make_binary_label_variables,
                                 make_binary_threshold_variables)
from torchfactors.components.at_least import KIsAtLeastJ
from torchfactors.subject import Environment
from torchfactors.testing import DummyParamNamespace, DummyVar


def test_make_binary_label_variables():
    a = tx.TensorVar(torch.tensor([0, 1, 2, 4]), tx.OBSERVED, tx.Range(5))
    b = DummyVar(4, domain_size=6)
    c = DummyVar(5, domain_size=2)
    env = Environment()
    vars = make_binary_label_variables(env, a, b, c, key='root')

    # shouldn't model the 0 value
    with pytest.raises(KeyError):
        vars[a, 0]
    assert vars[a, 1].tensor.tolist() == [0, 1, 1, 1]
    assert vars[a, 2].tensor.tolist() == [0, 0, 1, 1]
    assert vars[a, 3].tensor.tolist() == [0, 0, 0, 1]
    assert vars[a, 4].tensor.tolist() == [0, 0, 0, 1]
    assert len(vars) == (5 - 1) + (6 - 1) + (2 - 1)
    assert len(vars[a, 2].domain) == 2
    assert len(vars[c, 1].domain) == 2
    assert env.variable((a, 2, 'root')) == vars[a, 2]


def test_make_latent_binary_label_variables():
    a = DummyVar(3)
    env = Environment()
    vars = make_binary_label_variables(env, a, key='root', latent=True)
    assert (vars[a, 1].usage == tx.LATENT).all()


def test_make_equals_binary_label_variables():
    a = tx.TensorVar(torch.tensor([0, 1, 2, 4]), tx.OBSERVED, tx.Range(5))
    b = DummyVar(4, domain_size=6)
    c = DummyVar(5, domain_size=2)
    env = Environment()
    vars = make_binary_label_variables(env, a, b, c, key='root', only_equals=True)
    assert vars[a, 1].tensor.tolist() == [0, 1, 0, 0]
    assert vars[a, 2].tensor.tolist() == [0, 0, 1, 0]
    assert vars[a, 3].tensor.tolist() == [0, 0, 0, 0]
    assert vars[a, 4].tensor.tolist() == [0, 0, 0, 1]


def test_make_threshold_variables():
    a = tx.TensorVar(torch.tensor([3, 0, 2, 4, 1]), tx.OBSERVED, tx.Range(5))
    b = tx.TensorVar(torch.tensor([3, 0, 5, 2, 4, 1]), tx.OBSERVED, tx.Range(6))
    c = DummyVar(5, domain_size=2)
    env = Environment()
    vars = make_binary_threshold_variables(env, a, b, c, key='root')
    assert len(vars) == 3
    # posiitive a should be top two lables: 3 and 4
    assert vars[a].tensor.tolist() == [1, 0, 0, 1, 0]
    # posiitive b should be top three lables: 3, 4, and 5
    assert vars[b].tensor.tolist() == [1, 0, 1, 0, 1, 0]


def test_make_latent_threshold_variables():
    a = tx.TensorVar(torch.tensor([3, 0, 2, 4, 1]), tx.OBSERVED, tx.Range(5))
    b = tx.TensorVar(torch.tensor([3, 0, 5, 2, 4, 1]), tx.LATENT, tx.Range(6))
    c = tx.TensorVar(torch.tensor([3, 0, 5, 2, 4, 1]), tx.PADDING, tx.Range(6))
    env = Environment()
    vars_obs = make_binary_threshold_variables(env, a, b, c, key='root')
    assert (vars_obs[a].usage == tx.OBSERVED).all()
    assert (vars_obs[b].usage == tx.LATENT).all()
    assert (vars_obs[c].usage == tx.PADDING).all()
    env = Environment()
    vars_lat = make_binary_threshold_variables(env, a, b, c, key='root', latent=True)
    assert (vars_lat[a].usage == tx.LATENT).all()
    assert (vars_lat[b].usage == tx.LATENT).all()
    assert (vars_lat[c].usage == tx.PADDING).all()


def test_binary_scores_module():
    params = DummyParamNamespace()
    a = tx.TensorVar(torch.tensor([3, 0, 2, 4, 1]), tx.OBSERVED, tx.Range(5))
    b = tx.TensorVar(torch.tensor([3, 0, 5, 2, 4, 1]), tx.LATENT, tx.Range(6))
    c = tx.TensorVar(torch.tensor([3, 0, 5, 2, 4, 1]), tx.PADDING, tx.Range(6))
    input_a = torch.tensor([1, 2, 3]).float()
    module = BinaryScoresModule(params, [a, b, c], input_a)
    out: torch.Tensor = module(input_a)
    module2 = BinaryScoresModule(params, [a, b, c], input_a)
    out2: torch.Tensor = module2(input_a)
    assert out.allclose(out2)
    input_b = torch.tensor([3, 2, 1]).float()
    out3: torch.Tensor = module2(input_b)
    assert not out.allclose(out3)


test_binary_scores_module()
# from typing import Iterable

# import torch
# import torchfactors as tx


# @tx.dataclass
# class Seq(tx.Subject):
#     items: tx.Var = tx.VarField(tx.Range(5))


# class Chain(tx.Model[Seq]):
#     def __init__(self, clique_model: tx.CliqueModel):
#         super().__init__()
#         self.clique_model = clique_model

#     def factors(self, subject: Seq) -> Iterable[tx.Factor]:
#         for index in range(subject.items.shape[-1]):
#             yield from self.clique_model.factors(
#                 subject.environment, self.namespace('unaries'),
#                 subject.items[..., index])
#         length = subject.items.shape[-1]
#         for index in range(1, length):
#             yield from self.clique_model.factors(
#                 subject.environment, self.namespace('pairs'),
#                 subject.items[..., index - 1], subject.items[..., index])


# def test_prop_odds():
#     model = Chain()
#     x = Seq(tx.vtensor([1, 2, 3, 4, 5]),
#             usage=torch.tensor([tx.ANNOTATED, tx.LATENT, tx.CLAMPED, tx.OBSERVED, tx.PADDING])
#     factors=list(model(x))
#     assert len(factors) == 5 + 4

def testKIsAtLeastJ():
    a = tx.TensorVar(torch.tensor([3, 0, 2, 1, 3]), tx.ANNOTATED, tx.Range(4))
    a1 = tx.TensorVar(torch.zeros(5), tx.ANNOTATED, tx.Range(2))
    a3 = tx.TensorVar(torch.zeros(5), tx.ANNOTATED, tx.Range(2))
    factor1 = KIsAtLeastJ(a, a1, 1)
    out1 = factor1.dense
    assert out1.shape == (5, 4, 2)
    expected1 = torch.tensor(  # j = 1
        [
            [  # batch element = 0
                [1, 0],  # k = 0, (k not at least j, is at least)
                [0, 1],  # k = 1, (1 is not at least 1?, 1 is at least 1?)
                [0, 1],  # k = 2,
                [0, 1],  # k = 3
            ],
        ] * 5).float()
    assert out1.exp().allclose(expected1)

    expected3 = torch.tensor(  # j = 3
        [
            [  # batch element = 0
                [1, 0],  # k = 0, (k not at least j, is at least)
                [1, 0],  # k = 1, (1 is not at least 3?, 1 is at least 3?)
                [1, 0],  # k = 2,
                [0, 1],  # k = 3
            ],
        ] * 5).float()
    factor3 = KIsAtLeastJ(a, a3, 3)
    out3 = factor3.dense
    assert out3.shape == (5, 4, 2)
    assert out3.exp().allclose(expected3)

    # a = tx.TensorVar(torch.tensor([0, 1, 2, 4]), tx.OBSERVED, tx.Range(5))
    # a = tx.TensorVar(torch.tensor([0, 1, 2, 4]), tx.OBSERVED, tx.Range(5))
    # env = Environment()
    # vars = make_binary_label_variables(env, a, key='root')


testKIsAtLeastJ()
