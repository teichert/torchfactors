from __future__ import annotations

import dataclasses
import logging
import re
from collections import ChainMap
from dataclasses import dataclass
from typing import (Any, Dict, Generic, List, Mapping, Optional, Sized, Union,
                    cast)

import pytorch_lightning as pl
import torch
from torch.functional import Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm  # type: ignore

from torchfactors.inferencer import Inferencer
from torchfactors.inferencers.bp import BP

from .model import Model
from .model_inferencer import System
from .subject import ChainDataset, ListDataset, SubjectType
from .utils import Config, split_data

optimizers = dict(
    Adam=torch.optim.Adam,
    AdamW=torch.optim.AdamW,
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
    lr=ArgParseArg(float, 1.0, 'learning rate'),
    weight_decay=ArgParseArg(float, None, 'weight decay'),
)

default_inference_kwargs = dict(
    passes=ArgParseArg(int, None, 'number of times each bp message will be sent')
)

lit_init_names = dict(
    optimizer=ArgParseArg(str, default='Adam'),
    inferencer=ArgParseArg(str, default='BP'),
    in_model=ArgParseArg(str, default=None),
    # out_model=ArgParseArg(str, default='model.pt'),
    penalty_coeff=ArgParseArg(float, 1.0,
                              'multiplied by the exponentiated total KL from previous message'))


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
    train: Dataset[SubjectType] = dataclasses.field(default_factory=ListDataset[SubjectType])
    val: Dataset[SubjectType] = dataclasses.field(default_factory=ListDataset[SubjectType])
    dev: Dataset[SubjectType] = dataclasses.field(default_factory=ListDataset[SubjectType])
    test: Dataset[SubjectType] = dataclasses.field(default_factory=ListDataset[SubjectType])

    @property
    def train_length(self) -> int:
        return len(cast(Sized, self.train))

    @property
    def val_length(self) -> int:
        return len(cast(Sized, self.val))

    @property
    def test_length(self) -> int:
        return len(cast(Sized, self.test if self.test_mode else self.dev))

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

    def make_data_loader(self, examples: Dataset[SubjectType],
                         batch_size: int | None,
                         limit: int | None):
        if limit is not None and len(cast(Sized, examples)) > limit:
            raise ValueError("You tried to create a dataloader with a bigger number of examples "
                             "than what your limit says.  When you set e.g. `dm.train = data`, "
                             "did you forget to limit `data`?: e.g. "
                             "`dm.train = data[:dm.train_limit]")
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
        return max(1, len(cast(Sized, examples))) if out is None else out

    def train_dataloader(self) -> DataLoader[SubjectType] | List[DataLoader[SubjectType]]:
        return self.make_data_loader(self.train, batch_size=self.train_batch_size,
                                     limit=self.train_limit)

    def val_dataloader(self) -> DataLoader[SubjectType] | List[DataLoader[SubjectType]]:
        return self.make_data_loader(self.val, batch_size=self.val_batch_size,
                                     limit=self.val_limit)

    def test_dataloader(self) -> DataLoader[SubjectType] | List[DataLoader[SubjectType]]:
        if self.test_mode:
            return self.make_data_loader(self.test, batch_size=self.test_batch_size,
                                         limit=self.test_limit)
        else:
            return self.make_data_loader(self.dev, batch_size=self.test_batch_size,
                                         limit=self.test_limit)

    def split_val_from_train(self):
        r"""
        achieves the target val number or proportion by spliting off the first
        so many examples from the train data
        """
        if self.val_portion is not None or self.val_max_count is not None:
            self.val, self.train = split_data(self.train,
                                              portion=self.val_portion,
                                              count=self.val_limit,
                                              generator=torch.Generator().manual_seed(42))

    def add_val_to_train(self):
        r"""
        after calling this method, there will be no val data
        and the train data will start with the examples
        that were in val
        """
        self.train = ChainDataset([self.val, self.train])
        self.val = ListDataset[SubjectType]()


class LitSystem(pl.LightningModule, Generic[SubjectType]):
    r"""
    Base class representing a modeling/data/training/eval regime for
    a torchfactors system. The purpose is to avoid repeated boilerplate
    if you want to use lightning for a torchfactors system with bp
    inference.

    If more generality is needed, then a new base class could pull some of this
    up into it.

    """
    # @ classmethod
    # def get_arg(cls, key: str, args: Dict[str, Any], defaults: Mapping[str, Any]):
    #     if key in args:
    #         return args[key]
    #     else:
    #         return defaults[key]

    # @ classmethod
    # def set_arg(cls, dest: Dict[str, Any], key: str, args: Dict[str, Any],
    #             defaults: Mapping[str, Any]):
    #     dest[key] = cls.get_arg(key, args, defaults)

    # @ classmethod
    # def from_args(cls,
    #               model: Model[SubjectType],
    #               data: DataModule[SubjectType],
    #               args: Optional[Namespace] = None,
    #               defaults: Mapping[str, Any] | None = None,
    #               **kwargs
    #               ) -> LitSystem[SubjectType]:
    #     if args is None:
    #         args = argparse.Namespace()
    #     args_dict = {k: v for k, v in vars(args).items() if v is not None}
    #     base_kwargs: Dict[str, Any] = {}
    #     optimizer_kwargs = {k: v.default for k, v in default_optimizer_kwargs.items()
    #                         if v.default is not None}
    #     inference_kwargs = {k: v.default for k, v in default_inference_kwargs.items()
    #                         if v.default is not None}
    #     field_names = set(f.name for f in dataclasses.fields(data))
    #     if defaults is None:
    #         defaults = {}
    #     # NOTE: TODO: could check some prefix like _optimizer or _inference
    #     # to handle cases where the params are not predeclared or have conflicts
    #     for key in set(args_dict.keys()).union(defaults.keys()):
    #         if key in default_optimizer_kwargs:
    #             cls.set_arg(optimizer_kwargs, key, args_dict, defaults)
    #         elif key in default_inference_kwargs:
    #             cls.set_arg(inference_kwargs, key, args_dict, defaults)
    #         elif key in field_names:
    #             v = cls.get_arg(key, args_dict, defaults)
    #             setattr(data, key, v)
    #         elif key in lit_init_names:
    #             cls.set_arg(base_kwargs, key, args_dict, defaults)
    #     return cls(model=model, data=data,
    #                optimizer_kwargs=optimizer_kwargs,
    #                inference_kwargs=inference_kwargs,
    #                **base_kwargs, **kwargs)

    def __init__(self,
                 model: Model[SubjectType],
                 data: Optional[pl.LightningDataModule] = None,
                 penalty_coeff: float = 1.0,
                 optimizer: str = 'torch.optim.Adam',
                 inferencer: str = 'torchfactors.inferencers.bp.BP',
                 config: Config | None = None,
                 #  in_model: str | None = None,
                 #  out_model: str = 'model.pt',
                 #  optimizer_kwargs: Optional[Dict[str, Any]] = None,
                 #  inference_kwargs: Optional[Dict[str, Any]] = None,
                 **kwargs
                 ):
        super().__init__()
        if config is None:
            self.config = Config(defaults=kwargs)
        else:
            self.config = config
        # params = locals()
        self.model = model
        # self.in_model = in_model
        # self.out_model = out_model
        self.optimizer_name = optimizer
        self.data = data
        self.log_info: Dict[str, Any] = {}
        self.penalty_coeff = penalty_coeff

        # self.self_kwargs = {
        #     k: params[k] for k in lit_init_names
        # }
        # if optimizer_kwargs is None:
        #     optimizer_kwargs = {k: v.default for k, v in default_optimizer_kwargs.items()
        #                         if v.default is not None}
        # self.optimizer_kwargs = optimizer_kwargs

        # if inference_kwargs is None:
        #     inference_kwargs = {k: v.default for k, v in default_inference_kwargs.items()
        #                         if v.default is not None}
        # inferencer_cls = inferencers[inferencer]
        # self.inference_kwargs = inference_kwargs
        # self.inferencer = inferencer_cls(**inference_kwargs)
        self.inferencer = cast(Inferencer, self.config.create_from_name(inferencer))
        self.system: System[SubjectType] = System(self.model, self.inferencer)

        self.primed = False
        if self.data is not None:
            self.data.setup(None)
        # self._hparams = dict(**self.optimizer_kwargs, **self.inference_kwargs, **self.self_kwargs)
        self._hparams = self.config.dict

    def get_progress_bar_dict(self) -> Mapping[str, Union[int, str]]:  # type: ignore
        return ChainMap(
            self.log_info,
            super().get_progress_bar_dict())

    @property
    def hparams(self):
        return self._hparams

    def on_fit_end(self) -> None:
        return super().on_fit_end()

    def setup(self, stage=None) -> None:
        # logging.info(self.optimizer_kwargs)
        # logging.info(self.inference_kwargs)
        # logging.info(self.self_kwargs)
        if not self.primed:
            with torch.set_grad_enabled(False):
                logging.info("priming")
                self.system.prime(cast(DataLoader[SubjectType], self.train_dataloader()))
            self.primed = True

    def configure_optimizers(self) -> Optimizer:
        return cast(Optimizer, self.config.create_from_name(self.optimizer_name,
                                                            params=self.parameters()))

    def transfer_batch_to_device(self, _batch, device, dataloader_idx):
        batch: SubjectType = cast(SubjectType, _batch)
        on_device = batch.to_device(device)
        return on_device

    def training_loss(self, batch: SubjectType, data_name: str) -> Tensor:
        info = dict(status="initializing")
        with tqdm(total=7, desc="Training Eval (forward)",
                  postfix=info, delay=0.5, leave=False) as progress:
            def update(status: str):
                info['status'] = status
                progress.set_postfix(info)
                progress.update()
            update('clamping')
            batch.clamp_annotated()
            update('clamped marginals')
            logz_clamped = self.system.product_marginal(batch)
            update('unclamping')
            batch.unclamp_annotated()
            update('unclamped marginals (and penalty)')
            logz_free, penalty = self.system.partition_with_change(batch)
            update('loss')
            loss = (logz_free - logz_clamped).clamp_min(0).sum()
            update('penalty')
            penalty = penalty.sum()
            on_epoch = True if data_name == 'train' else None
            self.log(f'step.{data_name}.free-energy', loss, on_epoch=on_epoch)
            self.log(f'step.{data_name}.penalty', penalty, on_epoch=on_epoch)
            update('full loss')
            if self.penalty_coeff != 0:
                loss = loss + self.penalty_coeff * penalty.exp()
            self.log(f'step.{data_name}.loss', loss, on_epoch=on_epoch)
            update('done')
            return loss

    def training_step(self, *args, **kwargs) -> Union[torch._tensor.Tensor, Dict[str, Any]]:
        batch: SubjectType = cast(SubjectType, args[0])
        return self.training_loss(batch, 'train')

    def train_dataloader(self) -> DataLoader[SubjectType] | List[DataLoader[SubjectType]]:
        if self.data is None:
            raise TypeError("did not specify data")
        return cast(DataLoader[SubjectType], self.data.train_dataloader())

    def val_dataloader(self) -> DataLoader[SubjectType] | List[DataLoader[SubjectType]]:
        if self.data is None:
            raise TypeError("did not specify data")
        return cast(DataLoader[SubjectType], self.data.val_dataloader())

    def test_dataloader(self) -> DataLoader[SubjectType] | List[DataLoader[SubjectType]]:
        if self.data is None:
            raise TypeError("did not specify data")
        return cast(DataLoader[SubjectType], self.data.test_dataloader())

    # @ classmethod
    # def add_argparse_args(cls, parser: ArgumentParser):
    #     parser = pl.Trainer.add_argparse_args(parser)
    #     for key, value in itertools.chain(
    #             default_optimizer_kwargs.items(),
    #             default_inference_kwargs.items(),
    #             lit_init_names.items()):
    #         parser.add_argument(f'--{key}', type=value.type, help=value.help,
    #                             default=value.default)
    #     for field in dataclasses.fields(DataModule):
    #         field_type = get_type(field.type)  # type: ignore
    #         if field_type is not None:
    #             parser.add_argument(f'--{field.name}', type=field_type)
    #     return parser

    def forward(self, *args, **kwargs) -> SubjectType:
        return self.forward_(cast(SubjectType, args[0]))

    def forward_(self, x: SubjectType) -> SubjectType:
        return self.system.predict(x)

    @property
    def txdata(self) -> DataModule[SubjectType]:
        return cast(DataModule[SubjectType], self.data)

# class DataLoader(Generic[SubjectType]):
#     def __getitem__(self, ix: Any) -> SubjectType: ...
