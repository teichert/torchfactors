import torchfactors as tx
from torchfactors.components.linear_factor import LinearFactor
from torchfactors.learning import example_fit_model


def test_learning():
    @tx.dataclass
    class MySubject(tx.Subject):
        v: tx.Var = tx.VarField(tx.Range(2), tx.ANNOTATED)

    examples = [MySubject(tx.vtensor(0)), MySubject(tx.vtensor(1))]
    examples0 = [MySubject(tx.vtensor(0)), MySubject(tx.vtensor(0))]
    examples1 = [MySubject(tx.vtensor(1)), MySubject(tx.vtensor(1))]

    class MyModel(tx.Model[MySubject]):
        def factors(self, x: MySubject):
            yield LinearFactor(self.namespace('unary'), x.v)

    model = MyModel()
    num_steps = 0

    def each_step(loader, example):
        nonlocal num_steps
        num_steps += 1
    iters = 5
    system = example_fit_model(model, examples=examples0, each_step=each_step, iterations=iters)
    assert num_steps == iters * len(examples0)
    out = system.predict(MySubject.stack(examples))
    assert out.v.flatten().tolist() == [0, 0]

    num_counted = 0

    def each_epoch(loader):
        nonlocal num_counted
        num_counted += 1
    iters = 5

    system = example_fit_model(model, examples=examples1, each_epoch=each_epoch, iterations=iters)
    assert num_counted == iters
    out = system.predict(MySubject.stack(examples))
    assert out.v.flatten().tolist() == [1, 1]
