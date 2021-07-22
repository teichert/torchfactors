from argparse import ArgumentParser
from typing import Any, Dict, Generic, Optional, Union, cast

import pytorch_lightning as pl
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader

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


class LitSystem(pl.LightningModule, Generic[SubjectType]):
    r"""
    Base class representing a modeling/data/training/eval regime for
    a torchfactors system. The purpose is to avoid repeated boilerplate
    if you want to use lightning for a torchfactors system with bp
    inference.

    If more generality is needed, then a new base class could pull some of this
    up into it.

    """

    def __init__(self,
                 model: Model[SubjectType],
                 penalty_coeff: float = 1.0,
                 optimizer: str = 'Adam',
                 inferencer: str = 'BP',
                 optimizer_kwargs: Optional[Dict[str, Any]] = None,
                 inference_kwargs: Optional[Dict[str, Any]] = None,
                 argparse_args: Optional[Dict[str, Any]] = None,
                 ):
        super().__init__()
        # self.model = self.configure_model()
        self.model = model
        self.optimizer_name = optimizer
        self.log_info: Dict[str, Any] = {}
        self.penalty_coeff = penalty_coeff

        if optimizer_kwargs is None:
            self.optimizer_kwargs = {}
        else:
            self.optimizer_kwargs = dict(optimizer_kwargs)

        if inference_kwargs is None:
            inference_kwargs = {}
        else:
            inference_kwargs = dict(inference_kwargs)

        if argparse_args is not None:
            here

        if 'lr' not in self.optimizer_kwargs:
            self.optimizer_kwargs['lr'] = 1.0

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
