

import argparse

import pytest
import pytorch_lightning as pl
import torchfactors as tx
from torchfactors.components.linear_factor import LinearFactor
from torchfactors.lightning import DataModule


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


def test_lightning_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser = tx.LitSystem.add_argparse_args(parser)
    args = parser.parse_args("")
    assert args.bp_iters == 3
    assert args.batch_size == -1
    assert args.maxn is None


def test_datamodule():
    class TestData_v1_0(DataModule):
        def setup(self, stage=None):
            self.train = examples

    data = TestData_v1_0()
    data.setup()
    assert len(data.train_dataloader()) == 2
    assert len(data.val_dataloader()) == 0
    assert len(data.test_dataloader()) == 0
