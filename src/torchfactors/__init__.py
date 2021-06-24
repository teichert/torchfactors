from __future__ import annotations

from .components.linear_factor import LinearFactor
from .components.tensor_factor import TensorFactor
from .domain import Domain, Range, SeqDomain
from .factor import Factor
from .factor_graph import FactorGraph
from .inferencer import Inferencer
from .inferencers.bp import BP
from .model import Model
from .model_inferencer import ModelInferencer
from .strategies.bethe_graph import BetheGraph
from .subject import Subject
from .utils import ndarange
from .variable import TensorVar, Var, VarBranch, VarField, VarUsage

PADDING = VarUsage.PADDING
LATENT = VarUsage.LATENT
ANNOTATED = VarUsage.ANNOTATED
CLAMPED = VarUsage.CLAMPED
OBSERVED = VarUsage.OBSERVED
DEFAULT = VarUsage.DEFAULT

OPEN = Domain.OPEN

__all__ = [
    'Domain', 'SeqDomain', 'Range',
    'OPEN',
    'VarUsage', 'Var', 'VarBranch', 'VarField', 'TensorVar',
    'PADDING', 'LATENT', 'ANNOTATED', 'CLAMPED', 'OBSERVED', 'DEFAULT',
    'Subject',
    'Factor',
    'TensorFactor', 'LinearFactor',
    'Model',
    'FactorGraph',
    'ndarange',
    'BetheGraph',
    'Inferencer',
    'ModelInferencer',
    'BP',
]
