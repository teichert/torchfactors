
import torch
from torchfactors.variable import Var

from ..factor import Factor
from .tensor_factor import TensorFactor


def KIsAtLeastJ(k: Var, kIsAtLeastJ: Var, j: int) -> Factor:
    r"""
    For `isAtLeastJ == 1`, only has non-zero probability for `k >= j`.
    For `isAtLeastJ == 0`, only has non-zero probability for `k < j`.
    """
    n = len(k.domain)
    tensor = torch.stack([
        torch.cat([
            torch.zeros((j,)),
            torch.full((n - j,), float('-inf')),
        ]),
        torch.cat([
            torch.full((j,), float('-inf')),
            torch.zeros((n - j,)),
        ]),
    ])
    expanded = tensor.t()[None].expand(
        Factor.shape_from_variables([k, kIsAtLeastJ]))
    out = TensorFactor(k, kIsAtLeastJ, tensor=expanded)
    return out
