import logging
from typing import Iterable, Optional, Protocol, Sequence, Tuple, Type, TypeVar

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
                if log_info is not None:
                    t.set_postfix(**log_info, refresh=False)


class SystemRunner(Protocol):

    def __call__(self, system: System[SubjectType],
                 data_loader: DataLoader[SubjectType],
                 data: SubjectType): ...


def example_fit_model(model: Model[SubjectType], examples: Sequence[SubjectType],
                      iterations: int = 10,
                      each_step: Optional[SystemRunner] = None,
                      each_epoch: Optional[SystemRunner] = None,
                      batch_size: int = -1, penalty_coeff=1, passes=3,
                      log_info=None,
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
    system = System(model, BP(passes=passes))
    system.prime(data_loader)
    optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)
    # optimizer = torch.optim.LBFGS(model.parameters(), lr=lr)
    # optimizer = torch.optim.Rprop(model.parameters(), lr=lr)
    # optimizer = torch.optim.Rprop(model.parameters(), lr=lr)

    # iterations=1000, lr=0.1, passes=3, penalty_coeff=100)
    # bits.py | 316/1000 [01:28<03:11,  3.58it/s, combo=251, i=315, j=0, loss=6.07, penalty=0.894]
    # optimizer = optim.RAdam(model.parameters(), lr=lr)
    # optimizer = optim.AdaBound(model.parameters(), lr=lr)
    # # | 373/1000 [02:03<03:04,  3.40it/s, combo=402, i=372, j=0, loss=6.02, penalty=1.38]
    # optimizer = optim.AdaBound(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer, max_lr=lr, epochs=iterations // 3, steps_per_epoch=3)
    logging.info('staring training...')
    if log_info is None:
        log_info = {}
    data: SubjectType
    for i, j, data in tnested(data_loader, iterations, log_info=log_info):  # , mininterval=1.0):
        log_info['i'] = i
        log_info['j'] = j
        # logging.info(f'iteration: {i}')
        # logging.info('\tzeroing grad...')

        def closure():
            optimizer.zero_grad()
            # logging.info('\tclamped inference...')
            clamped = data.clamp_annotated()
            free = data.unclamp_annotated()
            logz_clamped = system.product_marginal(clamped)
            # for t in system.model(data):
            #     logging.info('\t\t\t---')
            #     for row in t.dense.tolist()[0]:
            #         logging.info(f'\t\t\t{row}')
            # logging.info(f'\t\tlogz_clamped: {logz_clamped}')
            # logging.info('\tunclamped inference...')
            logz_free, penalty = system.partition_with_change(free)
            # for t in system.model(data):
            #     logging.info('\t\t\t---')
            #     for row in t.dense.tolist()[0]:
            #         logging.info(f'\t\t\t{row}')
            # logging.info(f'\t\tlogz_free: {logz_free}')
            # logging.info(f'\t\tpenalty: {penalty}')
            loss = (logz_free - logz_clamped).clamp_min(0).sum()
            log_info['loss'] = float(loss)
            log_info['penalty'] = float(penalty)
            # logging.info(f'\t\tloss: {loss}')
            if penalty_coeff != 0:
                loss = loss + penalty_coeff * penalty.sum().exp()
            log_info['combo'] = float(loss)
            # logging.info(f'\t\tpenalty coeff: {penalty_coeff}')
            # logging.info(f'\t\tpenalized loss: {loss}')
            # logging.info('\tcomputing gradient...')
            loss.backward()
            # print([p.grad for p in system.model.parameters()])
            # logging.info('\tupdating...')
            return loss

        optimizer.step(closure)
        # log_info['lr'] = scheduler.get_last_lr()
        # scheduler.step()
        if each_step is not None:
            each_step(system, data_loader, data)
        if j == 0 and each_epoch is not None:
            each_epoch(system, data_loader, data)
    return system
