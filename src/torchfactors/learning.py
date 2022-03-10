import logging
from typing import (Iterable, Optional, Protocol, Sequence, Tuple, Type,
                    TypeVar, Union)

import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm  # type: ignore

from .inferencers.bp import BP
from .model import Model
from .model_inferencer import System
from .subject import SubjectType

T = TypeVar('T')


def nested(itr: Iterable[T], times: int) -> Iterable[Tuple[int, int, T]]:
    items = list(itr)
    for i in range(times):
        for j, obj in enumerate(items):
            yield i, j, obj


def tnested(itr: Iterable[T], times: int, log_info=None, leave=True, **kwargs
            ) -> Iterable[Tuple[int, int, T]]:
    items = list(itr)
    if log_info is None:
        yield from tqdm(nested(items, times), total=len(items) * times, leave=leave, **kwargs)
    else:
        with tqdm(nested(items, times), total=len(items) * times, leave=leave, **kwargs) as t:
            for i, j, obj in t:
                yield i, j, obj
                t.set_postfix(**log_info, refresh=False)


class SystemRunner(Protocol):

    def __call__(self, system: System[SubjectType],
                 data_loader: DataLoader[SubjectType],
                 data: SubjectType): ...  # pragma: no cover


def example_fit_model(model: Model[SubjectType], examples: Sequence[SubjectType],
                      iterations: int = 10,
                      each_step: Optional[SystemRunner] = None,
                      each_epoch: Optional[SystemRunner] = None,
                      batch_size: int = -1, penalty_coeff=1, passes=3,
                      log_info: Union[None, dict, str] = None,
                      inferencer=None,
                      optimizer_cls: Type[Optimizer] = torch.optim.Adam,
                      **optimizer_kwargs
                      ) -> System[SubjectType]:
    r"""
    Trains the model on the given training examples and returns the trained system.
    """
    if 'lr' not in optimizer_kwargs:
        optimizer_kwargs['lr'] = 1.0

    logging.info('loading...')
    data_loader = examples[0].data_loader(list(examples), batch_size=batch_size)
    logging.info('done loading.')
    if inferencer is None:
        inferencer = BP(passes=passes)
    system = System(model, inferencer)
    system.prime(data_loader)
    optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)
    logging.info('starting training...')
    if log_info is None:
        log_info = {}
    elif log_info == 'off':
        log_info = None
    data: SubjectType
    for i, j, data in tnested(data_loader, iterations, log_info=log_info):
        if isinstance(log_info, dict):
            log_info['i'] = i
            log_info['j'] = j

        def closure():
            optimizer.zero_grad()
            ll, change = system.log_likelihood_with_change(data)
            loss = -ll.sum()
            penalty = change.sum()
            if isinstance(log_info, dict):
                log_info['loss'] = float(loss)
                log_info['penalty'] = float(penalty)
            if penalty_coeff != 0:
                loss = loss + penalty_coeff * penalty.exp()
            if isinstance(log_info, dict):
                log_info['combo'] = float(loss)
            if loss.requires_grad:
                loss.backward()
            return loss

        optimizer.step(closure)
        if each_step is not None:
            each_step(system, data_loader, data)
        if j == 0 and each_epoch is not None:
            each_epoch(system, data_loader, data)
    return system
