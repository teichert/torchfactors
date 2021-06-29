from typing import Callable, Optional, Sequence

import torch
from torch.utils.data.dataloader import DataLoader

from .inferencers.bp import BP
from .model import Model
from .model_inferencer import System
from .subject import SubjectType


def example_fit_model(model: Model[SubjectType], examples: Sequence[SubjectType],
                      optimizer=torch.optim.Adam, iterations: int = 10,
                      each_step: Optional[Callable[[
                          DataLoader[SubjectType], SubjectType], None]] = None,
                      each_epoch: Optional[Callable[[DataLoader[SubjectType]], None]] = None,
                      lr=1.0) -> System[SubjectType]:
    data_loader = examples[0].data_loader(list(examples))
    system = System(model, BP())
    system.prime(data_loader)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(iterations):
        data: SubjectType
        for data in data_loader:
            optimizer.zero_grad()
            logz_clamped = system.product_marginal(data.clamp_annotated_())
            logz_free = system.product_marginal(data.unclamp_annotated_())
            loss = (logz_free - logz_clamped).sum()
            print(loss)
            loss.backward()
            optimizer.step()
            if each_step is not None:
                each_step(data_loader, data)
        if each_epoch is not None:
            each_epoch(data_loader)
    return system
