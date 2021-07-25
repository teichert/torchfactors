

import argparse
from typing import cast

import pytest
import pytorch_lightning as pl
import torchfactors as tx
from torchfactors.components.linear_factor import LinearFactor
from torchfactors.lightning import DataModule, get_type


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
    args = parser.parse_args(
        "--passes 5 --lr 10 --val_max_count 0 "
        "--batch_size 2 --test_batch_size 1".split())
    assert args.passes == 5
    assert args.lr == 10
    assert args.val_max_count == 0
    assert args.batch_size == 2
    assert args.test_batch_size == 1
    sys = tx.LitSystem.from_args(MyModel(), MyData_v1_0(),
                                 args)
    assert sys.inferencer.passes == 5
    assert sys.optimizer_kwargs['lr'] == 10
    assert len(sys.train_dataloader()) == 1
    assert len(sys.val_dataloader()) == 0
    assert len(sys.test_dataloader()) == 2


class MyData_v1_0(DataModule):

    def setup(self, stage=None):
        max_counts = self.split_max_counts(stage)
        for split, count in max_counts.items():
            self.set_split(split, examples[:count])


def test_datamodule():
    data = MyData_v1_0()
    data.setup()
    assert len(data.train_dataloader()) == 2
    assert len(data.val_dataloader()) == 2
    assert len(data.test_dataloader()) == 2


def test_datamodule2():
    data = MyData_v1_0(train_batch_size=2)
    data.setup()
    assert len(data.train_dataloader()) == 1
    assert len(data.val_dataloader()) == 2
    assert len(data.test_dataloader()) == 2


def test_datamodule3():
    data = MyData_v1_0(batch_size=2)
    data.setup()
    assert len(data.train_dataloader()) == 1
    assert len(data.val_dataloader()) == 1
    assert len(data.test_dataloader()) == 1


def test_datamodule4():
    data = MyData_v1_0(train_max_count=1)
    data.setup()
    assert len(data.train_dataloader()) == 1
    assert len(data.val_dataloader()) == 2
    assert len(data.test_dataloader()) == 2


def test_datamodule5():
    data = MyData_v1_0(split_max_count=1)
    data.setup()
    assert len(data.train_dataloader()) == 1
    assert len(data.val_dataloader()) == 1
    assert len(data.test_dataloader()) == 1


def test_datamodule7():
    data = MyData_v1_0()
    data.setup('fit')
    assert len(data.train_dataloader()) == 2
    assert len(data.val_dataloader()) == 2
    assert len(data.test_dataloader()) == 0


def test_datamodule8():
    data = MyData_v1_0()
    data.setup('test')
    assert len(data.train_dataloader()) == 0
    assert len(data.val_dataloader()) == 0
    assert len(data.test_dataloader()) == 2


# class MyLit(tx.lightning.LitSystem):


def test_args():
    model = MyModel()
    sys = tx.lightning.LitSystem.from_args(
        model=model, data=MyData_v1_0())
    assert len(sys.train_dataloader()) == 2
    assert len(sys.val_dataloader()) == 2
    assert len(sys.test_dataloader()) == 2


def test_args2():
    model = MyModel()
    sys = tx.lightning.LitSystem.from_args(
        model=model, data=MyData_v1_0(),
        defaults=dict(lr=5.0, optimizer='LBFGS',
                      path="hello!",
                      val_batch_size=2,
                      train_max_count=0,
                      passes=10,
                      fast_dev_run=True,
                      ))
    assert len(sys.train_dataloader()) == 0
    assert len(sys.val_dataloader()) == 1
    assert len(sys.test_dataloader()) == 2
    assert sys.optimizer_kwargs['lr'] == 5.0
    assert sys.optimizer_name == 'LBFGS'
    assert sys.inferencer.passes == 10
    assert cast(DataModule, sys.data).path == "hello!"


def test_args3():
    model = MyModel()
    namespace = argparse.Namespace()
    d = vars(namespace)
    d.update(dict(lr=5.0, optimizer='LBFGS',
                  path="hello!",
                  val_batch_size=2,
                  train_max_count=0,
                  passes=10,
                  fast_dev_run=True
                  ))
    sys = tx.lightning.LitSystem.from_args(
        model=model, data=MyData_v1_0(),
        args=namespace)
    assert len(sys.train_dataloader()) == 0
    assert len(sys.val_dataloader()) == 1
    assert len(sys.test_dataloader()) == 2
    assert sys.optimizer_kwargs['lr'] == 5.0
    assert sys.optimizer_name == 'LBFGS'
    assert sys.inferencer.passes == 10
    assert cast(DataModule, sys.data).path == "hello!"


def test_nodata():
    model = MyModel()
    sys = tx.lightning.LitSystem(model=model)

    with pytest.raises(TypeError):
        sys.train_dataloader()

    with pytest.raises(TypeError):
        sys.val_dataloader()

    with pytest.raises(TypeError):
        sys.test_dataloader()


def test_get_type():
    assert get_type("Optinoal[str]") is str
    assert get_type("str | int") is str
    assert get_type("str") is str
    assert get_type("int | str") is int
    assert get_type("blah") is None
    assert get_type("bool") is bool
    assert get_type("float") is float


test_lightning_args()
