from __future__ import annotations

import logging
from dataclasses import dataclass

from torch.nn.functional import one_hot

from . import learning
from .clique import CliqueModel
from .components.linear_factor import LinearFactor, ShapedLinear
from .components.tensor_factor import TensorFactor
from .domain import Domain, FlexDomain, Range, SeqDomain
from .einsum import log_dot
from .factor import Factor
from .factor_graph import FactorGraph
from .inferencer import Inferencer
from .inferencers.bp import BP
from .inferencers.brute_force import BruteForce
from .model import Model
from .model_inferencer import System
from .strategies.bethe_graph import BetheGraph
from .subject import Subject
from .types import GeneralizedDimensionDrop, gdrop
from .utils import data_len, logsumexp, ndarange, num_trainable
from .variable import (TensorVar, Var, VarBranch, VarField, VarUsage, at,
                       vtensor)

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

PADDING = VarUsage.PADDING
LATENT = VarUsage.LATENT
ANNOTATED = VarUsage.ANNOTATED
CLAMPED = VarUsage.CLAMPED
OBSERVED = VarUsage.OBSERVED
DEFAULT = VarUsage.DEFAULT

OPEN = Domain.OPEN

__all__ = [
    'Config',
    'ShapedLinear',
    'data_len',
    'gdrop', 'GeneralizedDimensionDrop', 'at', 'num_trainable',
    'dataclass', 'logsumexp',
    'one_hot',
    'CliqueModel',
    'Domain', 'SeqDomain', 'Range', 'FlexDomain',
    'OPEN',
    'VarUsage', 'Var', 'VarBranch', 'VarField', 'TensorVar', 'vtensor',
    'PADDING', 'LATENT', 'ANNOTATED', 'CLAMPED', 'OBSERVED', 'DEFAULT',
    'Subject',
    'Factor',
    'TensorFactor', 'LinearFactor',
    'Model',
    'FactorGraph',
    'ndarange',
    'BetheGraph',
    'Inferencer',
    'System',
    'BP',
    'BruteForce',
    'learning', 'log_dot'
]
