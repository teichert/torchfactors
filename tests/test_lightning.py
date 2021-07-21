

import pytest
import pytorch_lightning as pl
import torchfactors as tx
from torchfactors.components.linear_factor import LinearFactor


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
data_loader = MySubject.data_loader(examples)


@pytest.mark.filterwarnings('ignore::UserWarning')
def test_lit_learning():
    trainer = pl.Trainer(max_epochs=10)
    lit = tx.LitSystem(MyModel())
    trainer.fit(lit, data_loader)


@pytest.mark.filterwarnings('ignore::UserWarning')
def test_lit_learning_fit_twice():
    trainer = pl.Trainer(max_epochs=10)
    lit = tx.LitSystem(MyModel())
    trainer.fit(lit, data_loader)
    trainer.fit(lit, data_loader)


@pytest.mark.filterwarnings('ignore::UserWarning')
def test_lit_learning_with_no_penalty():
    trainer = pl.Trainer(max_epochs=10)
    lit = tx.LitSystem(MyModel(), penalty_coeff=0.0)
    trainer.fit(lit, data_loader)


@pytest.mark.filterwarnings('ignore::UserWarning')
def test_lit_learning_with_inference_kwargs():
    trainer = pl.Trainer(max_epochs=10)
    lit = tx.LitSystem(MyModel(), inference_kwargs=dict(passes=10))
    trainer.fit(lit, data_loader)


@pytest.mark.filterwarnings('ignore::UserWarning')
def test_lit_learning_with_optimizers_kwargs():
    trainer = pl.Trainer(max_epochs=10)
    lit = tx.LitSystem(MyModel(), optimizer_kwargs=dict(lr=0.01))
    trainer.fit(lit, data_loader)
