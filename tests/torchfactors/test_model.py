import math
import os
import tempfile
from dataclasses import dataclass
from typing import Iterable

import pytest
import torch
from pytest import approx
from torch import Tensor
from torch.nn import functional as F

import torchfactors as tx
from torchfactors import (BP, LATENT, Factor, Model, Range, Subject, System,
                          Var, VarField)
from torchfactors.components.tensor_factor import TensorFactor
from torchfactors.domain import FlexDomain
from torchfactors.inferencers.brute_force import BruteForce
from torchfactors.model import ParamNamespace
from torchfactors.testing import DummyParamNamespace, DummySubject
from torchfactors.variable import VarUsage


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

    # check that system.prime works
    system = System(model, BP())
    assert len(list(model.parameters())) == 0
    system.prime(Seq())
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


@tx.dataclass
class MySubject(tx.Subject):
    i: int = 1


def test_parameters2():
    m = Model[MySubject]()
    ns_a = m.namespace('a')
    with pytest.raises(KeyError):
        # no parameter with this key
        ns_a.parameter()
    ns_a_b = ns_a.namespace('b')
    assert ns_a.parameter((3, 3)).shape == (3, 3)
    assert ns_a_b.parameter((4, 6)).shape == (4, 6)
    ns_a.parameter()
    assert list(m(MySubject())) == []


def test_modules():
    m = Model[MySubject]()
    with pytest.raises(KeyError):
        # need to specify a factory if not already built for that key
        m.namespace('root').module()
    module = m.namespace('root').module(
        torch.nn.Linear, in_features=4, out_features=5, bias=False)
    with pytest.raises(KeyError):
        # this key has been used for a module rather than a param
        m.namespace('root').parameter()
    module2 = m.namespace('root').module()
    assert module is module2
    params = list(module2.parameters())
    assert len(params) == 1
    assert params[0].T.shape == (4, 5)
    module3 = tx.model.build_module('torch.nn.Linear', in_features=4, out_features=5, bias=False)
    params3 = list(module3.parameters())
    assert params3[0].T.shape == (4, 5)


def test_new_modules():
    m = Model[MySubject]()
    with pytest.raises(KeyError):
        # need to specify a factory if not already built for that key
        m.namespace('root').module()

    module = m.namespace('root').module(
        torch.nn.Linear, in_features=4, out_features=5, bias=False)

    with pytest.raises(KeyError):
        # this key has been used for a module rather than a param
        m.namespace('root').parameter()
    module2 = m.namespace('root').module()
    assert module is module2
    params = list(module2.parameters())
    assert len(params) == 1
    assert params[0].T.shape == (4, 5)


def test_model_inferencer():

    @dataclass
    class MySubject(Subject):
        items: Var = VarField(Range(5), LATENT, shape=(2,))

    class MyModel(Model[MySubject]):
        def factors(self, s: MySubject):
            first = s.items[..., 0]
            dom_size = len(first.domain)
            # the first one should be likely on 3
            yield TensorFactor(first,
                               tensor=F.one_hot(torch.tensor(3), dom_size).log())

            n = s.items.shape[-1]
            # all of them should be the same
            for i in range(n - 1):
                cur = s.items[..., i]
                next = s.items[..., i + 1]
                yield TensorFactor(cur, next, tensor=torch.eye(dom_size).log())

    system = System(model=MyModel(), inferencer=BP())
    subject = MySubject()
    out = system.predict(subject)
    assert (out.items.tensor == 3).all()

    # there is only one valid assignment which has a score of 0.0
    assert system.product_marginal(MySubject()) == 0.0
    x = MySubject()
    marg, = system.product_marginals(x, [x.items[..., 2]])
    assert (marg.exp() == torch.tensor([0, 0, 0, 1, 0])).all()


@dataclass
class MySubject_inferencer2(Subject):
    items: Var = VarField(Range(5), LATENT, shape=(7,))


class MyModel_inferencer2(Model[MySubject_inferencer2]):
    def factors(self, s: MySubject_inferencer2):
        first = s.items[..., 0]
        dom_size = len(first.domain)
        # the first one should be likely on 3
        yield TensorFactor(first,
                           tensor=(
                               F.one_hot(torch.tensor(3), dom_size) +
                               F.one_hot(torch.tensor(1), dom_size))
                           .log())
        n = s.items.shape[-1]
        # all of them should be the same
        for i in range(n - 1):
            cur = s.items[..., i]
            next = s.items[..., i + 1]
            yield TensorFactor(cur, next, tensor=torch.eye(dom_size).log())


def test_model_inferencer2():
    model = MyModel_inferencer2()
    system = System(model=model, inferencer=BP())
    out = system.predict(MySubject_inferencer2())
    out_t = out.items.tensor
    assert (out_t == 1).logical_or(out_t == 3).all()

    # there are only 2 options since they all need to be the same
    logz = system.product_marginal(MySubject_inferencer2())
    assert float(logz) == approx(math.log(2))

    x = MySubject_inferencer2()
    marg, = system.product_marginals(x, [x.items[..., 2]])
    assert (marg.exp() == torch.tensor([0, 0.5, 0, 0.5, 0])).all()

    subj = MySubject_inferencer2()
    subj.items.set_usage(VarUsage.ANNOTATED)
    subj.items.tensor = torch.tensor(3)
    assert system.log_likelihood(subj).exp() == approx(0.5)


def test_model_inferencer2_brute():
    model = MyModel_inferencer2()
    system = System(model=model, inferencer=BruteForce())
    out = system.predict(MySubject_inferencer2())
    out_t = out.items.tensor
    assert (out_t == 1).logical_or(out_t == 3).all()


@dataclass
class Characters(tx.Subject):
    char: tx.Var = tx.VarField(tx.Range(255), tx.ANNOTATED)

    @property
    def view(self) -> str:
        return ''.join(map(chr, self.char.tensor.tolist()))

    @staticmethod
    def from_string(text: str) -> 'Characters':
        text_nums = list(map(ord, text))
        return Characters(tx.TensorVar(torch.tensor(text_nums)))


class Unigrams(tx.Model[Characters]):
    def factors(self, x: Characters):
        yield tx.LinearFactor(self.namespace('unigram'), x.char)


def test_prime_linear_params():
    model = Unigrams()
    system = tx.System(model, tx.BP())
    assert len(list(model.parameters())) == 0
    system.prime(Characters.from_string('testing'))
    assert len(list(model.parameters())) > 0


def test_custom_init():
    params = DummyParamNamespace()
    constant = 3.0

    def my_init(t: Tensor):
        t[(...,)] = constant

    out = params.parameter((3, 4), initialization=my_init)
    assert out.allclose(torch.ones(3, 4) * constant)
    out2 = params.parameter()
    assert out2 is out


def test_model_domain_state_dict():
    model = Model[DummySubject]()
    domain = FlexDomain('test', unk=True)
    values1 = ['this', 'test', 'is', 'a', 'test', 'of', 'this']
    values2 = ['test', 'a', 'test', 'of', 'this', 'is', 'this']
    ids1 = model.domain_ids(domain, values1).tolist()
    assert ids1 == [1, 2, 3, 4, 2, 5, 1]
    ids2 = model.domain_ids(domain, values2).tolist()
    assert ids2 == [2, 4, 2, 5, 1, 3, 1]
    state = model.state_dict()
    with tempfile.TemporaryDirectory() as test_dir:
        path = os.path.join(test_dir, '__test_model.pt')
        torch.save(state, path)
        loaded_state = torch.load(path)
        model2 = Model[DummySubject]()
        model2.load_state_dict(loaded_state)

        domain2 = FlexDomain('test', unk=True)
        out2 = model2.domain_ids(domain2, values2).tolist()
        assert out2 == ids2
        out1 = model2.domain_ids(domain2, values1).tolist()
        assert out1 == ids1


def test_model_domain_state_dict_with_params():
    model = Model[DummySubject]()
    domain = FlexDomain('test', unk=True)
    values1 = ['this', 'test', 'is', 'a', 'test', 'of', 'this']
    values2 = ['test', 'a', 'test', 'of', 'this', 'is', 'this']
    ids1 = model.domain_ids(domain, values1).tolist()
    assert ids1 == [1, 2, 3, 4, 2, 5, 1]
    ids2 = model.domain_ids(domain, values2).tolist()
    assert ids2 == [2, 4, 2, 5, 1, 3, 1]

    def init(t: Tensor):
        t[(...,)] = 3
    params = model.namespace('hi').parameter((3, 5), init)
    paramsb = model.namespace('hi').parameter()
    assert (paramsb == params).all()
    state = model.state_dict()

    with tempfile.TemporaryDirectory() as test_dir:
        path = os.path.join(test_dir, '__test_model.pt')
        torch.save(state, path)
        loaded_state = torch.load(path)
        model2 = Model[DummySubject]()
        model2.load_state_dict(loaded_state)
        params2 = model2.namespace('hi').parameter()
        assert (params2 == params).all()
        assert (params2 == torch.ones(3, 5) * 3).all()

        domain2 = FlexDomain('test', unk=True)
        out2 = model2.domain_ids(domain2, values2).tolist()
        assert out2 == ids2
        out1 = model2.domain_ids(domain2, values1).tolist()
        assert out1 == ids1

        domain3: FlexDomain = model2.domain('testing')
        assert domain3.get_id('test') == 0
        assert domain3.get_id('test2') == 1
        assert domain3.get_id('test') == 0

        domain4: FlexDomain = model2.domain('testing')
        assert domain4.get_id('test2') == 1
        assert domain4.get_id('test') == 0


def test_model_domain_state_dict_with_params_and_modules():
    model = Model[DummySubject]()
    domain = FlexDomain('test', unk=True)
    values1 = ['this', 'test', 'is', 'a', 'test', 'of', 'this']
    values2 = ['test', 'a', 'test', 'of', 'this', 'is', 'this']
    ids1 = model.domain_ids(domain, values1).tolist()
    assert ids1 == [1, 2, 3, 4, 2, 5, 1]
    ids2 = model.domain_ids(domain, values2).tolist()
    assert ids2 == [2, 4, 2, 5, 1, 3, 1]

    def init(t: Tensor):
        t[(...,)] = 3
    params = model.namespace('hi').parameter((3, 5), init)
    module = model.namespace('hi2').module(
        tx.ShapedLinear, output_shape=(3, 2), bias=False, input_shape=params.shape)
    out_from_module = module(params)
    assert out_from_module.shape == (3, 2)
    paramsb = model.namespace('hi').parameter()
    assert (paramsb == params).all()
    state = model.state_dict()
    with tempfile.TemporaryDirectory() as test_dir:
        path = os.path.join(test_dir, '__test_model.pt')
        torch.save(state, path)
        loaded_state = torch.load(path)
        model2 = Model[DummySubject]()
        model2.load_state_dict(loaded_state)
        params2 = model2.namespace('hi').parameter()
        assert (params2 == params).all()
        assert (params2 == torch.ones(3, 5) * 3).all()
        module2 = model2.namespace('hi2').module()
        out_from_module2 = module2(params2)
        assert (out_from_module2 == out_from_module).all()
        domain2 = FlexDomain('test', unk=True)
        out2 = model2.domain_ids(domain2, values2).tolist()
        assert out2 == ids2
        out1 = model2.domain_ids(domain2, values1).tolist()
        assert out1 == ids1


def test_load_model1():
    model = Model[DummySubject]()
    domain = FlexDomain('test', unk=True)
    values1 = ['this', 'test', 'is', 'a', 'test', 'of', 'this']
    values2 = ['test', 'a', 'test', 'of', 'this', 'is', 'this']
    ids1 = model.domain_ids(domain, values1).tolist()
    assert ids1 == [1, 2, 3, 4, 2, 5, 1]
    ids2 = model.domain_ids(domain, values2).tolist()
    assert ids2 == [2, 4, 2, 5, 1, 3, 1]

    def init(t: Tensor):
        t[(...,)] = 3
    params = model.namespace('hi').parameter((3, 5), init)
    module = model.namespace('hi2').module(
        tx.ShapedLinear, output_shape=(3, 2), bias=False, input_shape=params.shape)
    out_from_module = module(params)
    assert out_from_module.shape == (3, 2)
    paramsb = model.namespace('hi').parameter()
    assert (paramsb == params).all()
    state = model.state_dict()
    with tempfile.TemporaryDirectory() as test_dir:
        path = os.path.join(test_dir, '__test_model.pt')
        torch.save(state, path)
        model2 = Model[DummySubject](model_state_dict_path=path)
        params2 = model2.namespace('hi').parameter()
        assert (params2 == params).all()
        assert (params2 == torch.ones(3, 5) * 3).all()
        module2 = model2.namespace('hi2').module()
        out_from_module2 = module2(params2)
        assert (out_from_module2 == out_from_module).all()
        domain2 = FlexDomain('test', unk=True)
        out2 = model2.domain_ids(domain2, values2).tolist()
        assert out2 == ids2
        out1 = model2.domain_ids(domain2, values1).tolist()
        assert out1 == ids1


def test_load_model2():
    model = Model[DummySubject]()
    domain = FlexDomain('test', unk=True)
    values1 = ['this', 'test', 'is', 'a', 'test', 'of', 'this']
    values2 = ['test', 'a', 'test', 'of', 'this', 'is', 'this']
    ids1 = model.domain_ids(domain, values1).tolist()
    assert ids1 == [1, 2, 3, 4, 2, 5, 1]
    ids2 = model.domain_ids(domain, values2).tolist()
    assert ids2 == [2, 4, 2, 5, 1, 3, 1]

    def init(t: Tensor):
        t[(...,)] = 3
    params = model.namespace('hi').parameter((3, 5), init)
    module = model.namespace('hi2').module(
        tx.ShapedLinear, output_shape=(3, 2), bias=False, input_shape=params.shape)
    out_from_module = module(params)
    assert out_from_module.shape == (3, 2)
    paramsb = model.namespace('hi').parameter()
    assert (paramsb == params).all()
    state = model.state_dict()
    with tempfile.TemporaryDirectory() as test_dir:
        path = os.path.join(test_dir, '__test_model2.pt')
        torch.save(dict(state_dict=state), path)
        model2 = Model[DummySubject](checkpoint_path=path)
        params2 = model2.namespace('hi').parameter()
        assert (params2 == params).all()
        assert (params2 == torch.ones(3, 5) * 3).all()
        module2 = model2.namespace('hi2').module()
        out_from_module2 = module2(params2)
        assert (out_from_module2 == out_from_module).all()
        domain2 = FlexDomain('test', unk=True)
        out2 = model2.domain_ids(domain2, values2).tolist()
        assert out2 == ids2
        out1 = model2.domain_ids(domain2, values1).tolist()
        assert out1 == ids1
