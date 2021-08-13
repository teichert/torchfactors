

import argparse
import tempfile
from typing import cast

import pytest
import pytorch_lightning as pl
import torch
import torchfactors as tx
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from torch.functional import Tensor
from torchfactors.components.linear_factor import LinearFactor
from torchfactors.domain import FlexDomain
from torchfactors.lightning import DataModule, get_type
from torchfactors.model import Model
from torchfactors.subject import ListDataset
from torchfactors.testing import DummySubject


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


more_examples = examples * 2


class MyData_v1_0(DataModule[MySubject]):

    def setup(self, stage=None):
        if stage in (None, 'fit'):
            self.train = ListDataset(more_examples[:self.train_limit])
            self.split_val_from_train()
        if stage in (None, 'test'):
            self.test = ListDataset(more_examples[:self.test_limit])
            self.dev = ListDataset(more_examples[:self.test_limit])


class MyData_v3_0(DataModule[MySubject]):

    def setup(self, stage=None):
        if stage in (None, 'fit'):
            self.train = ListDataset(more_examples[:self.train_limit])
            self.val = ListDataset(more_examples[:self.val_limit])
            self.add_val_to_train()
            self.split_val_from_train()
        if stage in (None, 'test'):
            self.test = ListDataset(more_examples[:self.test_limit])
            self.dev = ListDataset(more_examples[:self.test_limit])


def test_lightning_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser = tx.LitSystem.add_argparse_args(parser)
    args = parser.parse_args(
        "--test_mode True --passes 5 --lr 10 --val_max_count 0 "
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
    assert len(sys.train_dataloader()) == 2
    assert len(sys.val_dataloader()) == 0
    assert len(sys.test_dataloader()) == 4


def test_datamodule():
    data = MyData_v1_0(test_mode=True)
    data.setup()
    assert len(data.train_dataloader()) == 4
    assert len(data.val_dataloader()) == 0
    assert len(data.test_dataloader()) == 4


def test_datamodule2():
    data = MyData_v1_0(train_batch_size=2)
    data.setup()
    assert len(data.train_dataloader()) == 2
    assert len(data.val_dataloader()) == 0
    assert len(data.test_dataloader()) == 4


def test_datamodule3():
    data = MyData_v1_0(batch_size=2)
    data.setup()
    assert len(data.train_dataloader()) == 2
    assert len(data.val_dataloader()) == 0
    assert len(data.test_dataloader()) == 2
    assert data.train_length == 4
    assert data.val_length == 0
    assert data.test_length == 4


def test_datamodule4():
    data = MyData_v1_0(train_max_count=1)
    data.setup()
    assert len(data.train_dataloader()) == 1
    assert len(data.val_dataloader()) == 0
    assert len(data.test_dataloader()) == 4


def test_datamodule5():
    data = MyData_v1_0(split_max_count=1)
    data.setup()
    assert len(data.train_dataloader()) == 1
    assert len(data.val_dataloader()) == 0
    assert len(data.test_dataloader()) == 1


def test_datamodule7():
    data = MyData_v1_0(val_portion=0.25)
    data.setup('fit')
    assert len(data.train_dataloader()) == 3
    assert len(data.val_dataloader()) == 1
    assert len(data.test_dataloader()) == 0
    assert data.train_length == 3
    assert data.val_length == 1
    assert data.test_length == 0


def test_datamodule8():
    data = MyData_v1_0()
    data.setup('test')
    assert len(data.train_dataloader()) == 0
    assert len(data.val_dataloader()) == 0
    assert len(data.test_dataloader()) == 4
    assert data.train_length == 0
    assert data.val_length == 0
    assert data.test_length == 4


def test_datamodule10():
    data = MyData_v3_0(val_portion=0.25, test_mode=True)
    data.setup(None)
    assert len(data.train_dataloader()) == 6
    assert len(data.val_dataloader()) == 2
    assert len(data.test_dataloader()) == 4
    assert data.train_length == 6
    assert data.val_length == 2
    assert data.test_length == 4


def test_datamodule9():
    class MyData_v2_0(tx.DataModule):
        def setup(self, stage=None):
            self.train_max_count = 1
            self.train = tx.ListDataset(examples)

    data = MyData_v2_0()
    data.setup()
    with pytest.raises(ValueError):
        # shouldn't be allowed to have 2 examples with a max_count of 1
        data.train_dataloader()


def test_args():
    model = MyModel()
    sys = tx.lightning.LitSystem.from_args(
        model=model, data=MyData_v1_0())
    assert len(sys.train_dataloader()) == 4
    assert len(sys.val_dataloader()) == 0
    assert len(sys.test_dataloader()) == 4


def test_args2():
    model = MyModel()
    sys = tx.lightning.LitSystem.from_args(
        model=model, data=MyData_v1_0(),
        defaults=dict(lr=5.0, optimizer='LBFGS',
                      test_mode=True,
                      path="hello!",
                      val_batch_size=2,
                      val_max_count=1,
                      val_portion=1.0,
                      passes=10,
                      fast_dev_run=True,
                      ))
    assert len(sys.train_dataloader()) == 3
    assert len(sys.val_dataloader()) == 1
    assert len(sys.test_dataloader()) == 4
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
                  test_mode=True,
                  val_batch_size=2,
                  val_max_count=1,
                  val_portion=1.0,
                  passes=10,
                  fast_dev_run=True
                  ))
    sys = tx.lightning.LitSystem.from_args(
        model=model, data=MyData_v1_0(),
        args=namespace)
    assert len(sys.train_dataloader()) == 3
    assert len(sys.val_dataloader()) == 1
    assert len(sys.test_dataloader()) == 4
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


def test_args4():
    model = MyModel()
    namespace = argparse.Namespace()
    d = vars(namespace)
    d.update(dict(lr=5.0, optimizer='LBFGS',
                  path="hello!",
                  batch_size=-1,
                  test_mode=True,
                  val_max_count=2,
                  passes=10,
                  fast_dev_run=True
                  ))
    sys = tx.lightning.LitSystem.from_args(
        model=model, data=MyData_v1_0(),
        args=namespace)
    assert len(sys.train_dataloader()) == 1
    assert len(sys.val_dataloader()) == 1
    assert len(sys.test_dataloader()) == 1
    assert sys.optimizer_kwargs['lr'] == 5.0
    assert sys.optimizer_name == 'LBFGS'
    assert sys.inferencer.passes == 10
    assert sys.txdata.path == "hello!"


def test_dev_test():
    @tx.dataclass
    class MySubject(tx.Subject):
        i: int

    class Data(tx.DataModule[MySubject]):
        def setup(self, stage=None):
            self.train = tx.ListDataset([MySubject(i) for i in range(0, 1)])
            self.val = tx.ListDataset([MySubject(i) for i in range(1, 2)])
            self.dev = tx.ListDataset([MySubject(i) for i in range(3, 6)])
            self.test = tx.ListDataset([MySubject(i) for i in range(9, 12)])

    data1 = Data(test_mode=False)
    data1.setup()
    assert [x.i for x in list(data1.test_dataloader())] == [3, 4, 5]

    data2 = Data(test_mode=True)
    data2.setup()
    assert [x.i for x in list(data2.test_dataloader())] == [9, 10, 11]

    data3 = Data()  # dev should be default
    data3.setup()
    assert [x.i for x in list(data3.test_dataloader())] == [3, 4, 5]


# def test_config():
#     class A:
#         def __init__(self, something, a=3, b='hi', c=True):
#             self.something = something
#             self.a = a
#             self.b = b
#             self.c = c


#     @tx.dataclass
#     class C:
#         a: str
#         i: int = 9


#     config = tx.Config(A, C)
#     assert config.parser.argumen


@pytest.mark.filterwarnings("ignore")
def test_load_model3():
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
    system = tx.LitSystem(model, tx.DataModule(train=[DummySubject()]))
    trainer = pl.Trainer(system, max_epochs=0)
    with tempfile.TemporaryDirectory() as tb_dir:
        trainer.logger = TensorBoardLogger(tb_dir)
        trainer.fit(system)
    checkpoint_path = '__test_checkpoint.ckpt'
    trainer.save_checkpoint(checkpoint_path)
    model2 = Model[DummySubject](checkpoint_path=checkpoint_path)
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
