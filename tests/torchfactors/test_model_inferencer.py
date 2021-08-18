import torchfactors as tx
from torchfactors.subject import Subject


def test_prime():
    @tx.dataclass
    class MySubject(Subject):
        i: int = 5

    examples = [MySubject(), MySubject(), MySubject()]
    counts = 0

    class MyModel(tx.Model[MySubject]):
        def factors(self, x: MySubject):
            nonlocal counts
            counts += 1
            return []

    model = MyModel()
    system = tx.System(model, tx.BP())
    system.prime(examples)
    assert counts == 3
