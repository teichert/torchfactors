from __future__ import annotations

import argparse
import dataclasses
import itertools
import re
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, Sized, Union, cast

import pytorch_lightning as pl
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from torchfactors.inferencers.bp import BP

from .model import Model
from .model_inferencer import System
from .subject import ListDataset, SubjectType
from .utils import split_data

optimizers = dict(
    Adam=torch.optim.Adam,
    LBFGS=torch.optim.LBFGS,
)

inferencers = dict(
    BP=BP,
)

typename_to_type = dict(
    str=str,
    int=int,
    float=float,
    bool=bool,
)


def get_type(typename: str) -> type | None:
    pieces = re.split('[^a-z]+', typename)
    for piece in pieces:
        try:
            return typename_to_type[piece]
        except KeyError:
            pass
    return None


@dataclass
class ArgParseArg:
    type: Any = str
    default: Any = None
    help: str = ""


default_optimizer_kwargs = dict(
    lr=ArgParseArg(float, 1.0, 'learning rate')
)

default_inference_kwargs = dict(
    passes=ArgParseArg(int, None, 'number of times each bp message will be sent')
)


lit_init_names = dict(
    optimizer=ArgParseArg(str, default='Adam'),
    inferencer=ArgParseArg(str, default='BP'),
    penalty_coeff=ArgParseArg(float, 1.0,
                              'multiplied by the exponentiated total KL from previous message'))


# class DataConsumer(Generic[SubjectType]):

#     def maybe_add(self, split: str, item: Callable[[], SubjectType]):
#         """
#         This should be overriden in every sub-class"""
#         raise NotImplementedError("data consumer not implemented")

#     def accept_item(self, split: str, item: SubjectType) -> bool:
#         return self.accept(split, split, lambda: item)

#     def accept(self, split: str, item: Callable[[], SubjectType]) -> bool:
#         try:
#             self.maybe_add(split, item)
#             return True
#         except StopIteration:
#             return False


@dataclass
class DataModule(pl.LightningDataModule, Generic[SubjectType]):
    r"""
    batch_size and max_count are general settings for all stages and
    will be overriden by more specific settings:
    -1 means no limit; None means not set;


    train data will be potentially limited and then split
    into train according to the smaller of ceil(len(filtered train) * val_portion))
    and val_max_count
    """
    path: str = ""
    split_max_count: int = -1
    batch_size: int = 1

    train_max_count: Optional[int] = None
    val_max_count: Optional[int] = None
    test_max_count: Optional[int] = None
    val_portion: Optional[float] = None

    train_batch_size: Optional[int] = None
    val_batch_size: Optional[int] = None
    test_batch_size: Optional[int] = None

    test_mode: bool = False
    train: Dataset[SubjectType] = dataclasses.field(default_factory=ListDataset)
    val: Dataset[SubjectType] = dataclasses.field(default_factory=ListDataset)
    test: Dataset[SubjectType] = dataclasses.field(default_factory=ListDataset)

    # def maybe_add(self, split: str, get: Callable[[], SubjectType]):
    #     if split == 'train':
    #         train = cast(ListDataset, self.train)
    #         if len(train.examples) < self.compute_max_count(self.train_max_count,
    #                                                         self.split_max_count):
    #             train.examples.append(get)
    #         else:
    #             raise StopIteration
    #     else:
    #         if split == 'dev' and self.test_mode:

    #         self.test_length < self.max_test_count and (
    #             self.test_mode and split == 'test' or
    #                 not self.test_mode and split == 'dev'):

    #     if split == 'train' and self.train_length < self.max_train_count:
    #         cast(ListDataset, self.train).examples.append(get())
    #     elif self.test_length < self.max_test_count and (
    #             self.test_mode and split == 'test' or
    #             not self.test_mode and split == 'dev'):
    #         cast(ListDataset, self.test).examples.append(get())
    #     else:
    #         raise StopIteration()

    @property
    def train_length(self) -> int:
        return len(cast(Sized, self.train))

    @property
    def val_length(self) -> int:
        return len(cast(Sized, self.val))

    @property
    def test_length(self) -> int:
        return len(cast(Sized, self.test))

    @property
    def train_limit(self) -> int:
        return self.compute_max_count(self.train_max_count)

    @property
    def val_limit(self) -> int:
        return self.compute_max_count(self.val_max_count)

    @property
    def test_limit(self) -> int:
        return self.compute_max_count(self.test_max_count)

    def __post_init__(self):
        super().__init__()

    @ staticmethod
    def negative_to_none(value: int) -> Optional[int]:
        return None if value < 0 else value

    def compute_max_count(self, split_max_count: Optional[int]):
        if split_max_count is None:
            return self.negative_to_none(self.split_max_count)
        else:
            return self.negative_to_none(split_max_count)

    # def set_split(self, split: str, examples: Dataset[SubjectType]
    #               ) -> None:
    #     dataset = cast(torch.utils.data.Dataset, examples)
    #     if split == 'train':
    #         self.train = dataset
    #     if split == 'val':
    #         self.val = dataset
    #     if split == 'test':
    #         self.test = dataset

    # def split_max_counts(self, stage: Optional[str]) -> Dict[str, int]:
    #     split_max_sizes = {}
    #     if stage in (None, 'fit'):
    #         split_max_sizes['train'] = self.compute_max_count(self.train_max_count)
    #     if stage in (None, 'test'):
    #         max_count = self.compute_max_count(self.test_max_count)
    #         if self.test_mode:
    #             split_max_sizes['test'] = max_count
    #         else:
    #             split_max_sizes['dev'] = max_count
    #     return split_max_sizes

    # def train(self) -> Sequence[SubjectType]:
    #     raise ValueError("no train data specified")

    def make_data_loader(self, examples: Dataset[SubjectType],
                         batch_size: int | None):
        computed_batch_size = self.computed_batch_size(examples, batch_size)
        if examples:
            return examples[0].data_loader(examples, batch_size=computed_batch_size)
        else:
            return DataLoader[SubjectType](examples, batch_size=computed_batch_size)

    def computed_batch_size(self, examples: Dataset[SubjectType],
                            split_batch_size: int | None):
        if split_batch_size is None:
            out = self.negative_to_none(self.batch_size)
        else:
            out = self.negative_to_none(split_batch_size)
        return len(cast(Sized, examples)) if out is None else out

    def train_dataloader(self) -> DataLoader | List[DataLoader]:
        return self.make_data_loader(self.train, batch_size=self.train_batch_size)

    def val_dataloader(self) -> DataLoader | List[DataLoader]:
        return self.make_data_loader(self.val, batch_size=self.val_batch_size)

    def test_dataloader(self) -> DataLoader | List[DataLoader]:
        return self.make_data_loader(self.test, batch_size=self.test_batch_size)

    def setup_val(self):
        if self.val_portion is not None or self.val_max_count is not None:
            self.val, self.train = split_data(self.train,
                                              portion=self.val_portion,
                                              count=self.val_limit,
                                              generator=torch.Generator().manual_seed(42))


class LitSystem(pl.LightningModule, Generic[SubjectType]):
    r"""
    Base class representing a modeling/data/training/eval regime for
    a torchfactors system. The purpose is to avoid repeated boilerplate
    if you want to use lightning for a torchfactors system with bp
    inference.

    If more generality is needed, then a new base class could pull some of this
    up into it.

    """
    @ classmethod
    def get_arg(cls, key: str, args: Dict[str, Any], defaults: Dict[str, Any]):
        if key in args:
            return args[key]
        else:
            return defaults[key]

    @ classmethod
    def set_arg(cls, dest: Dict[str, Any], key: str, args: Dict[str, Any],
                defaults: Dict[str, Any]):
        dest[key] = cls.get_arg(key, args, defaults)

    @ classmethod
    def from_args(cls,
                  model: Model[SubjectType],
                  data: DataModule[SubjectType],
                  args: Optional[Namespace] = None,
                  defaults: Dict[str, Any] | None = None,
                  **kwargs
                  ) -> LitSystem[SubjectType]:
        if args is None:
            args = argparse.Namespace()
        args_dict = {k: v for k, v in vars(args).items() if v is not None}
        base_kwargs: Dict[str, Any] = {}
        optimizer_kwargs = {k: v.default for k, v in default_optimizer_kwargs.items()
                            if v.default is not None}
        inference_kwargs = {k: v.default for k, v in default_inference_kwargs.items()
                            if v.default is not None}
        field_names = set(f.name for f in dataclasses.fields(data))
        if defaults is None:
            defaults = {}
        # NOTE: TODO: could check some prefix like _optimizer or _inference
        # to handle cases where the params are not predeclared or have conflicts
        for key in set(args_dict.keys()).union(defaults.keys()):
            if key in default_optimizer_kwargs:
                cls.set_arg(optimizer_kwargs, key, args_dict, defaults)
            elif key in default_inference_kwargs:
                cls.set_arg(inference_kwargs, key, args_dict, defaults)
            elif key in field_names:
                v = cls.get_arg(key, args_dict, defaults)
                setattr(data, key, v)
            elif key in lit_init_names:
                cls.set_arg(base_kwargs, key, args_dict, defaults)
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
            optimizer_kwargs = {k: v.default for k, v in default_optimizer_kwargs.items()
                                if v.default is not None}
        self.optimizer_kwargs = optimizer_kwargs

        if inference_kwargs is None:
            inference_kwargs = {k: v.default for k, v in default_inference_kwargs.items()
                                if v.default is not None}
        inferencer_cls = inferencers[inferencer]
        self.inferencer = inferencer_cls(**inference_kwargs)

        self.system: System[SubjectType] = System(
            self.model, self.inferencer)

        self.primed = False
        if self.data is not None:
            self.data.setup()

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

    @ classmethod
    def add_argparse_args(cls, parser: ArgumentParser):
        parser = pl.Trainer.add_argparse_args(parser)
        for key, value in itertools.chain(
                default_optimizer_kwargs.items(),
                default_inference_kwargs.items(),
                lit_init_names.items()):
            parser.add_argument(f'--{key}', type=value.type, help=value.help,
                                default=value.default)
        for field in dataclasses.fields(DataModule):
            field_type = get_type(field.type)  # type: ignore
            if field_type is not None:
                parser.add_argument(f'--{field.name}', type=field_type)
        return parser
