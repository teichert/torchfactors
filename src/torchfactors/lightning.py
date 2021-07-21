from abc import abstractmethod
from typing import Generic

import pytorch_lightning as pl
import torch
from torch.optim import Optimizer

from torchfactors.inferencer import Inferencer
from torchfactors.model import Model
from torchfactors.subject import SubjectType

from .model_inferencer import System


class LitSystem(pl.LightningModule, Generic[SubjectType]):

    def __init__(self,
                 penalty_coeff: float = 1.0,
                 **optim_kwargs
                 ):
        super().__init__()
        self.model = self.configure_model()
        self.inferencer = self.configure_inferencer()
        self.system: System[SubjectType] = System(
            self.model, self.inferencer)
        self.log_info = {}
        self.penalty_coeff = penalty_coeff
        if 'lr' not in optim_kwargs:
            optim_kwargs['lr'] = 1.0
        self.optim_kwargs = optim_kwargs

    @abstractmethod
    def configure_model(self) -> Model[SubjectType]: ...

    def configure_optimizers(self) -> Optimizer:
        self.system.prime(self.train_dataloader())
        return torch.optim.Adam(self.parameters(), **self.optim_kwargs)

    @abstractmethod
    def configure_inferencer(self) -> Inferencer: ...

    def forward(self, x: SubjectType) -> SubjectType:
        return self.system.predict(x)

    def training_step(self, batch: SubjectType, batch_idx: int):
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
