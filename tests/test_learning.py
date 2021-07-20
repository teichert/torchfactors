
import torchfactors as tx
from torchfactors.components.linear_factor import LinearFactor
from torchfactors.learning import example_fit_model, tnested


def test_learning():
    @tx.dataclass
    class MySubject(tx.Subject):
        v: tx.Var = tx.VarField(tx.Range(2), tx.ANNOTATED)

    examples = [MySubject(tx.vtensor(0)), MySubject(tx.vtensor(1))]
    examples0 = [MySubject(tx.vtensor(0)), MySubject(tx.vtensor(0))]
    examples1 = [MySubject(tx.vtensor(1)), MySubject(tx.vtensor(1))]
    stacked_examples = MySubject.stack(examples)

    class MyModel(tx.Model[MySubject]):
        def factors(self, x: MySubject):
            yield LinearFactor(self.namespace('unary'), x.v)

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

    system = example_fit_model(model, examples=examples1, each_epoch=each_epoch, iterations=iters,
                               batch_size=-1, log_info='off')
    out = system.predict(stacked_examples)
    assert out.v.flatten().tolist() == [1, 1]


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


test_tnested()
