from dataclasses import dataclass

import torch

from .domain import Range
from .model import Model
from .subject import Subject
from .variable import TensorVar, VarUsage


@dataclass
class DummySubject(Subject):
    i: int
    # v: Var = VarField(Range(5), VarUsage.OBSERVED)


def DummyParamNamespace():
    model = Model[DummySubject]()
    return model.namespace('root')


def DummyVar(*shape: int, domain_size: int = 5):
    return TensorVar(torch.ones(shape), VarUsage.OBSERVED, Range(domain_size))
