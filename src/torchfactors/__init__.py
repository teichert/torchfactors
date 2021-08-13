from __future__ import annotations

import logging
from dataclasses import dataclass

from torch.nn.functional import one_hot

from . import skip_some_warnings  # noqa
from . import learning
from .clique import CliqueModel
from .components.linear_factor import LinearFactor, ShapedLinear
from .components.tensor_factor import TensorFactor
from .domain import Domain, FlexDomain, Range, SeqDomain
from .einsum import log_dot
from .extra_utils import with_rep_number
from .factor import Factor
from .factor_graph import FactorGraph
from .inferencer import Inferencer
from .inferencers.bp import BP
from .lightning import DataModule, ListDataset, LitSystem
from .model import Model
from .model_inferencer import System
from .strategies.bethe_graph import BetheGraph
from .subject import Subject
from .types import GeneralizedDimensionDrop, gdrop
from .utils import Config, data_len, logsumexp, ndarange, num_trainable
from .variable import TensorVar, Var, VarField, VarUsage, at, vtensor

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
    'ListDataset', 'DataModule', 'LitSystem', 'with_rep_number',
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
    'learning', 'log_dot'
]
