from __future__ import annotations

import dataclasses
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, Sequence, Union, cast

import pytorch_lightning as pl
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from torchfactors.inferencers.bp import BP

from .model import Model
from .model_inferencer import System
from .subject import SubjectType

optimizers = dict(
    Adam=torch.optim.Adam,
    LBFGS=torch.optim.LBFGS,
)

inferencers = dict(
    BP=BP,
)

default_optimizer_kwargs = dict(
    lr=1.0,
)

default_inference_kwargs = dict(
    passes=None,
)


@dataclass
class DataModule(pl.LightningDataModule, Generic[SubjectType]):
    r"""
    batch_size and max_count are general settings for all stages and
    will be overriden by more specific settings:
    -1 means no limit; None means not set
    """
    path: str = ""
    split_max_count: int = -1
    batch_size: int = -1

    train_max_count: Optional[int] = None
    val_max_count: Optional[int] = None
    test_max_count: Optional[int] = None

    train_batch_size: Optional[int] = None
    val_batch_size: Optional[int] = None
    test_batch_size: Optional[int] = None

    train: Dataset[SubjectType] | Sequence[SubjectType] = cast(Sequence[SubjectType], ())
    val: Dataset[SubjectType] | Sequence[SubjectType] = cast(Sequence[SubjectType], ())
    test: Dataset[SubjectType] | Sequence[SubjectType] = cast(Sequence[SubjectType], ())

    def __post_init__(self):
        super().__init__()

    @staticmethod
    def negative_to_none(value: int) -> Optional[int]:
        return None if value < 0 else value

    def compute_max_count(self, split_max_count: Optional[int]):
        if split_max_count is None:
            return self.negative_to_none(self.split_max_count)
        else:
            return self.negative_to_none(split_max_count)

    def set_split(self, split: str, examples: Dataset[SubjectType] | Sequence[SubjectType]
                  ) -> None:
        dataset = cast(torch.utils.data.Dataset, examples)
        if split == 'train':
            self.train = dataset
        if split == 'val':
            self.val = dataset
        if split == 'val':
            self.val = dataset

    def split_max_counts(self, stage: Optional[str]) -> Dict[str, int]:
        split_max_sizes = {}
        if stage in (None, 'train'):
            split_max_sizes['train'] = self.compute_max_count(self.train_max_count)
        if stage in (None, 'val'):
            split_max_sizes['val'] = self.compute_max_count(self.val_max_count)
        if stage in (None, 'test'):
            split_max_sizes['test'] = self.compute_max_count(self.test_max_count)
        return split_max_sizes

    # def train(self) -> Sequence[SubjectType]:
    #     raise ValueError("no train data specified")

    def make_data_loader(self, examples: Dataset[SubjectType] | Sequence[SubjectType],
                         batch_size: int | None):
        computed_batch_size = self.computed_batch_size(batch_size)
        if examples:
            return examples[0].data_loader(examples, batch_size=computed_batch_size)
        else:
            return DataLoader[SubjectType](cast(Dataset[SubjectType], examples),
                                           batch_size=computed_batch_size)

    def computed_batch_size(self, split_batch_size: int | None):
        if split_batch_size is None:
            return self.negative_to_none(self.batch_size)
        else:
            return self.negative_to_none(split_batch_size)

    def train_dataloader(self) -> DataLoader | List[DataLoader]:
        return self.make_data_loader(self.train, batch_size=self.train_batch_size)

    def val_dataloader(self) -> DataLoader | List[DataLoader]:
        return self.make_data_loader(self.val, batch_size=self.val_batch_size)

    def test_dataloader(self) -> DataLoader | List[DataLoader]:
        return self.make_data_loader(self.test, batch_size=self.test_batch_size)


class LitSystem(pl.LightningModule, Generic[SubjectType]):
    r"""
    Base class representing a modeling/data/training/eval regime for
    a torchfactors system. The purpose is to avoid repeated boilerplate
    if you want to use lightning for a torchfactors system with bp
    inference.

    If more generality is needed, then a new base class could pull some of this
    up into it.

    """
    @classmethod
    def get_arg(cls, key: str, args: Namespace, defaults: Dict[str, Any]):
        if key in args:
            return getattr(args, key)
        else:
            return defaults[key]

    @classmethod
    def set_arg(cls, dest: Dict[str, Any], key: str, args: Namespace, defaults: Dict[str, Any]):
        dest[key] = cls.get_arg(key, args, defaults)

    @classmethod
    def from_argparse_args(cls, args: Namespace,
                           model: Model[SubjectType],
                           data: DataModule[SubjectType],
                           defaults: Dict[str, Any] | None = None,
                           **kwargs
                           ) -> LitSystem[SubjectType]:
        base_kwargs: Dict[str, Any] = {}
        optimizer_kwargs = {k: v for k, v in default_optimizer_kwargs.items() if v is not None}
        inference_kwargs = {k: v for k, v in default_inference_kwargs.items() if v is not None}
        field_names = set(f.name for f in dataclasses.fields(data))
        init_names = {'optimizer', 'inferencer', 'penalty_coeff'}
        if defaults is None:
            defaults = {}
        # NOTE: TODO: could check some prefix like _optimizer or _inference
        # to handle cases where the params are not predeclared or have conflicts
        for key in set(args.keys()).union(defaults.keys()):
            if key in default_optimizer_kwargs:
                cls.set_arg(optimizer_kwargs, key, args, defaults)
            elif key in default_inference_kwargs:
                cls.set_arg(default_inference_kwargs, key, args, defaults)
            elif key in field_names:
                v = cls.get_arg(key, args, defaults)
                setattr(data, key, v)
            elif key in init_names:
                cls.set_arg(base_kwargs, key, args, defaults)
        return cls(model=model, data=data,
                   optimizer_kwargs=optimizer_kwargs,
                   inference_kwargs=inference_kwargs,
                   **base_kwargs, **kwargs)

    def __init__(self,
                 model: Model[SubjectType],
                 data: Optional[pl.LightningDataModule] = None,
                 penalty_coeff: float = 1.0,
                 optimizer: str = 'Adam',
                 inferencer: str = 'BP',
                 optimizer_kwargs: Optional[Dict[str, Any]] = None,
                 inference_kwargs: Optional[Dict[str, Any]] = None,
                 ):
        super().__init__()
        # self.model = self.configure_model()
        self.model = model
        self.optimizer_name = optimizer
        self.data = data
        self.log_info: Dict[str, Any] = {}
        self.penalty_coeff = penalty_coeff

        if optimizer_kwargs is None:
            optimizer_kwargs = {k: v for k, v in default_optimizer_kwargs.items() if v is not None}
        self.optimizer_kwargs = optimizer_kwargs

        if inference_kwargs is None:
            inference_kwargs = {k: v for k, v in default_inference_kwargs.items() if v is not None}
        inferencer_cls = inferencers[inferencer]
        self.inferencer = inferencer_cls(**inference_kwargs)

        self.system: System[SubjectType] = System(
            self.model, self.inferencer)

        self.primed = False

    # @abstractmethod
    # def configure_model(self) -> Model[SubjectType]: ...

    def setup(self, stage=None) -> None:
        if not self.primed:
            with torch.set_grad_enabled(False):
                self.system.prime(cast(DataLoader[SubjectType], self.train_dataloader()))
            self.primed = True

    def configure_optimizers(self) -> Optimizer:
        return optimizers[self.optimizer_name](self.parameters(), **self.optimizer_kwargs)

    # @abstractmethod
    # def configure_inferencer(self) -> Inferencer: ...

    # def forward(self, x: SubjectType, *args, **kwargs) -> SubjectType:
    #     return self.system.predict(x)
    def transfer_batch_to_device(self, _batch, device):
        batch: SubjectType = cast(SubjectType, _batch)
        return batch.to_device(device)

    def training_step(self, *args, **kwargs) -> Union[torch._tensor.Tensor, Dict[str, Any]]:
        batch: SubjectType = cast(SubjectType, args[0])
        batch_idx: int = cast(int, args[1])
        self.log('batch_idx', batch_idx)
        clamped = batch.clamp_annotated()
        free = batch.unclamp_annotated()
        logz_clamped = self.system.product_marginal(clamped)
        logz_free, penalty = self.system.partition_with_change(free)
        loss = (logz_free - logz_clamped).clamp_min(0).sum()
        penalty = penalty.sum()
        self.log('loss', loss)
        self.log('penalty', penalty)
        if self.penalty_coeff != 0:
            loss = loss + self.penalty_coeff * penalty.exp()
        self.log('combo', loss)
        return loss

    def train_dataloader(self):
        if self.data is None:
            raise TypeError("did not specify data")
        return self.data.train_dataloader()

    def val_dataloader(self):
        if self.data is None:
            raise TypeError("did not specify data")
        return self.data.val_dataloader()

    def test_dataloader(self):
        if self.data is None:
            raise TypeError("did not specify data")
        return self.data.test_dataloader()

    @staticmethod
    def add_argparse_args(parser: ArgumentParser):
        parser = pl.Trainer.add_argparse_args(parser)
        parser.add_argument('--lr', type=float, default=0.1,
                            help="learning rate")
        parser.add_argument('--optimizer', type=str, default="LBFGS",
                            help=f"name of optimizer from {list(optimizers.keys())}")
        parser.add_argument('--bp_iters', type=int, default=3,
                            help="number of times each message will be sent in bp")
        parser.add_argument('--batch_size', type=int, default=-1,
                            help="size of each batch (-1 for all in single batch)")
        parser.add_argument('--maxn', type=int, default=None,
                            help="the maximum number of examples that will be included")
        return parser
