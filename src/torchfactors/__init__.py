from __future__ import annotations

from .components.linear_factor import LinearFactor
from .components.tensor_factor import TensorFactor
from .domain import Domain, Range, SeqDomain
from .factor import DensableFactor, Factor
from .factor_graph import FactorGraph
from .infer import query
from .model import Model
from .subject import Subject
from .variable import Var, VarBase, VarBranch, VarUsage

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
    'VarUsage', 'VarBase', 'Var', 'VarBranch',
    'PADDING', 'LATENT', 'ANNOTATED', 'CLAMPED', 'OBSERVED', 'DEFAULT',
    'Subject',
    'Factor', 'DensableFactor',
    'TensorFactor',
    'LinearFactor',
    'Model',
    'FactorGraph',
    'query',
]
