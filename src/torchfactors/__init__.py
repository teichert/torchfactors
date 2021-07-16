from __future__ import annotations

import logging
from dataclasses import dataclass

from . import learning
from .clique import CliqueModel
from .components.linear_factor import LinearFactor
from .components.tensor_factor import TensorFactor
from .domain import Domain, FlexDomain, Range, SeqDomain
from .einsum import ids, log_dot
from .factor import Factor
from .factor_graph import FactorGraph
from .inferencer import Inferencer
from .inferencers.bp import BP
from .model import Model
from .model_inferencer import System
from .strategies.bethe_graph import BetheGraph
from .subject import Subject
from .utils import logsumexp, ndarange
from .variable import TensorVar, Var, VarField, VarUsage, vtensor

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

PADDING = VarUsage.PADDING
LATENT = VarUsage.LATENT
ANNOTATED = VarUsage.ANNOTATED
CLAMPED = VarUsage.CLAMPED
OBSERVED = VarUsage.OBSERVED
DEFAULT = VarUsage.DEFAULT

OPEN = Domain.OPEN

__all__ = [
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
    'learning', 'log_dot', 'ids'
]
