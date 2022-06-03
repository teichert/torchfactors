
import math
from typing import Dict

import torch
from pytest import approx

import torchfactors as tx
from torchfactors.components.linear_factor import LinearFactor
from torchfactors.inferencers.brute_force import BruteForce
from torchfactors.learning import example_fit_model, tnested


@tx.dataclass
class MySubject(tx.Subject):
    v: tx.Var = tx.VarField(tx.Range(2), tx.ANNOTATED)


class MyModel(tx.Model[MySubject]):
    def factors(self, x: MySubject):
        yield LinearFactor(self.namespace('unary'), x.v)


examples = [MySubject(tx.vtensor(0)), MySubject(tx.vtensor(1))]
examples0 = [MySubject(tx.vtensor(0)), MySubject(tx.vtensor(0))]
examples1 = [MySubject(tx.vtensor(1)), MySubject(tx.vtensor(1))]
stacked_examples = MySubject.stack(examples)


def test_learning():
    model = MyModel()
    num_steps = 0

    def each_step(system, loader, example):
        nonlocal num_steps
        num_steps += 1
    iters = 5
    system = example_fit_model(model, examples=examples0, each_step=each_step, iterations=iters,
                               batch_size=1)
    assert num_steps == iters * len(examples0)
    out = system.predict(stacked_examples)
    assert out.v.flatten().tolist() == [0, 0]


def test_ilearning():
    model = MyModel()
    num_steps = 0
    iters = 5

    def each_step(system, loader, example):
        nonlocal num_steps
        num_steps += 1

    def criteria(i):
        return i < iters

    system = example_fit_model(model, examples=examples0, each_step=each_step, criteria=criteria,
                               batch_size=1)
    assert num_steps == iters * len(examples0)
    out = system.predict(stacked_examples)
    assert out.v.flatten().tolist() == [0, 0]


def test_learning2():
    model = MyModel()
    num_counted = 0

    def each_epoch(system, loader, example):
        nonlocal num_counted
        num_counted += 1
    iters = 5

    system = example_fit_model(model, examples=examples1, each_epoch=each_epoch, iterations=iters,
                               batch_size=1)
    assert num_counted == iters
    out = system.predict(stacked_examples)
    assert out.v.flatten().tolist() == [1, 1]


def test_learning3():
    model = MyModel()
    system = example_fit_model(model, examples=examples1, iterations=5,
                               batch_size=-1, log_info='off')
    out = system.predict(stacked_examples)
    assert out.v.flatten().tolist() == [1, 1]


def test_learning3i():
    model = MyModel()
    num_counted = 0
    iters = 5

    def each_epoch(system, loader, example):
        nonlocal num_counted
        num_counted += 1

    system = example_fit_model(model, examples=examples1, criteria=lambda i: i < iters,
                               batch_size=-1, log_info='off', each_epoch=each_epoch)
    out = system.predict(stacked_examples)
    assert num_counted == iters
    assert out.v.flatten().tolist() == [1, 1]


def test_learning4():
    model = MyModel()
    system = example_fit_model(model, examples=examples1, iterations=5,
                               batch_size=-1, lr=1.0)
    out = system.predict(stacked_examples)
    assert out.v.flatten().tolist() == [1, 1]


def test_learning4_lbfgs():
    model = MyModel()
    system = example_fit_model(model, examples=examples1, iterations=5,
                               batch_size=-1, lr=1.0, optimizer_cls=torch.optim.LBFGS,
                               line_search_fn='strong_wolfe')
    out = system.predict(stacked_examples)
    assert out.v.flatten().tolist() == [1, 1]


def test_learning4_nograd():
    # exercises the code that runs if grad is not enabled during step
    out = []

    class MyOptimizer(torch.optim.Optimizer):
        def __init__(self, params, **kwargs):
            super().__init__(params, {})
            self.params = params
            self.kwargs = kwargs

        def step(self, closure):
            with torch.set_grad_enabled(False):
                loss = closure()
                out.append(1)
                return loss

    model = MyModel()
    iterations = 5
    example_fit_model(model, examples=examples1, iterations=iterations,
                      batch_size=1, lr=1.0, optimizer_cls=MyOptimizer)
    assert out == [1] * iterations * 2


def test_learning4_with_penalty():
    model = MyModel()
    log_info: Dict[str, float] = {}
    coeff = 10.0
    example_fit_model(model, examples=examples1, iterations=5,
                      batch_size=-1, lr=1.0, penalty_coeff=coeff,
                      log_info=log_info)
    assert log_info['combo'] == approx(
        log_info['loss'] + coeff * math.exp(log_info['penalty']))


def test_learning4_without_penalty():
    model = MyModel()
    log_info: Dict[str, float] = {}
    example_fit_model(model, examples=examples1, iterations=5,
                      batch_size=-1, lr=1.0, penalty_coeff=0.0,
                      log_info=log_info)
    assert float(log_info['combo']) == float(log_info['loss'])


def test_learning5():
    model = MyModel()
    log_info: Dict[str, float] = {}
    system = example_fit_model(model, examples=examples1, iterations=5,
                               batch_size=-1, log_info=log_info)
    out = system.predict(stacked_examples)
    assert out.v.flatten().tolist() == [1, 1]
    assert set(log_info.keys()) == {'i', 'j', 'loss', 'penalty', 'combo'}


def test_learning5_bruteforce():
    model = MyModel()
    log_info: Dict[str, float] = {}
    system = example_fit_model(model, examples=examples1, iterations=5,
                               batch_size=-1, log_info=log_info, inferencer=BruteForce())
    out = system.predict(stacked_examples)
    assert out.v.flatten().tolist() == [1, 1]
    assert set(log_info.keys()) == {'i', 'j', 'loss', 'penalty', 'combo'}


def test_learning6():
    model = MyModel()
    gets = []
    sets = []

    class MyDict(dict):
        def __getitem__(self, k):
            gets.append(k)
            return super().__getitem__(k)

        def __setitem__(self, k, v):
            sets.append(k)
            super().__setitem__(k, v)
    log_info = MyDict()
    num_iters = 5
    system = example_fit_model(model, examples=examples1, iterations=num_iters,
                               batch_size=-1, log_info=log_info)
    out = system.predict(stacked_examples)
    assert out.v.flatten().tolist() == [1, 1]
    assert set(log_info.keys()) == {'i', 'j', 'loss', 'penalty', 'combo'}
    assert sets == ['i', 'j', 'loss', 'penalty', 'combo'] * num_iters


def test_tnested():
    out = list(tnested([3, 4, 2], 4))
    expected = [
        (0, 0, 3),
        (0, 1, 4),
        (0, 2, 2),
        (1, 0, 3),
        (1, 1, 4),
        (1, 2, 2),
        (2, 0, 3),
        (2, 1, 4),
        (2, 2, 2),
        (3, 0, 3),
        (3, 1, 4),
        (3, 2, 2),
    ]
    assert out == expected
    # try it with log-info off
    out2 = list(tnested([3, 4, 2], 4, log_info=None))
    assert out2 == expected
