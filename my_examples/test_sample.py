import torchfactors as tx

from sprl import SPR, SPR1DataModule


def test_SPR():
    class DummyModel(tx.Model[SPR]):
        def factors(self, x: SPR):
            return
    model = DummyModel()
    SPR1DataModule(model=model)


def test_add():
    assert 3 + 5 == 8
