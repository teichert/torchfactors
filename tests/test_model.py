from dataclasses import dataclass
from typing import Any, Iterable

import pytest
import torch
from torchfactors import (LATENT, Factor, Model, Range, Subject, Var, VarField)
from torchfactors.components.tensor_factor import TensorFactor
from torchfactors.model import ParamNamespace


@dataclass
class Seq(Subject):
    items: Var = VarField(Range(5), LATENT, shape=(4,))


class Chain(Model[Seq]):
    def factors(self, subject: Seq) -> Iterable[Factor]:
        for index in range(subject.items.shape[-1]):
            yield TensorFactor(subject.items[..., index])
        for index in range(subject.items.shape[-1] - 1):
            yield TensorFactor(subject.items[..., index], subject.items[..., index + 1])


def test_model():
    model = Chain()
    data = Seq()
    factors = list(model(data))
    assert len(factors) == 4 + 3


class MyBiasFactor(TensorFactor):
    def __init__(self, params: ParamNamespace, *variables: Var):
        super().__init__(
            *variables,
            tensor=params.parameter(
                Factor.out_shape_from_variables(variables)
            ).expand(
                Factor.shape_from_variables(variables)
            ))


def test_parameters():
    class Chain2(Model[Seq]):
        def factors(self, subject: Seq) -> Iterable[Factor]:
            for index in range(subject.items.shape[-1]):
                yield MyBiasFactor(self.namespace('emission'),
                                   subject.items[..., index])
            for index in range(subject.items.shape[-1] - 1):
                yield MyBiasFactor(self.namespace('transition'),
                                   subject.items[..., index], subject.items[..., index + 1])

    model = Chain2()
    list(model(Seq()))
    assert len(list(model.parameters())) == 2
    with pytest.raises(KeyError):
        # cannot get transition as a module since it was used as a parameter
        model.namespace('transition').module()

    assert model.namespace('emission').parameter().shape == (5,)
    assert (model.namespace('emission').parameter() == 0.0).all()
    assert model.namespace('transition').parameter().shape == (5, 5)
    assert (model.namespace('transition').parameter() != 0.0).all()
    with pytest.raises(KeyError):
        # has different shape than before
        model.namespace('transition').parameter((2, 10))


def test_parameters2():
    m = Model[Any]()
    ns_a = m.namespace('a')
    with pytest.raises(KeyError):
        # no parameter with this key
        ns_a.parameter()
    ns_a_b = ns_a.namespace('b')
    assert ns_a.parameter((3, 3)).shape == (3, 3)
    assert ns_a_b.parameter((4, 6)).shape == (4, 6)
    ns_a.parameter()
    assert list(m(object())) == []


def test_modules():
    m = Model[Any]()
    with pytest.raises(KeyError):
        # need to specify a factory if not already built for that key
        m.namespace('root').module()

    module = m.namespace('root').module(lambda: torch.nn.Linear(4, 5, bias=False))
    with pytest.raises(KeyError):
        # this key has been used for a module rather than a param
        m.namespace('root').parameter()
    module2 = m.namespace('root').module()
    assert module is module2
    params = list(module2.parameters())
    assert len(params) == 1
    assert params[0].T.shape == (4, 5)


# def test_model_inferencer():
#     system = System(model=Chain(), inferencer=BP())
#     out = system.predict(Seq())
