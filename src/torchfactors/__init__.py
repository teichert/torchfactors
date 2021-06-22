from __future__ import annotations

from .components.linear_factor import LinearFactor
from .components.tensor_factor import TensorFactor
from .domain import Domain, Range, SeqDomain
from .factor import DensableFactor, Factor
from .factor_graph import FactorGraph
from .infer import marginals
from .model import Model
from .subject import Subject
from .utils import ndrange
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
    'Factor', 'DensableFactor',
    'TensorFactor',
    'LinearFactor',
    'Model',
    'FactorGraph',
    'marginals',
    'ndrange'
]
