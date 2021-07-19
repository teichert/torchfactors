
import torch
from torchfactors.variable import Var

from ..factor import Factor
from .tensor_factor import TensorFactor


def KIsAtLeastJ(k: Var, kIsAtLeastJ: Var, j: int) -> Factor:
    r"""
    For `isAtLeastJ == 1`, only has non-zero probability for `k >= j`.
    For `isAtLeastJ == 0`, only has non-zero probability for `k < j`.
    """

    # def __init__(self, ):
    #     super().__init__([ordinal, *binaries])
    #     self.ordinal = ordinal
    #     self.binaries = binaries

    # def dense_(self) -> Tensor:
    #     """
    #     there will be n configs with non-zero prob:
    #     all zeros
    #     1
    #     """
    # n = len(ordinal.domain)
    # non_zero_tuples = [
    #     [i, *[int(i >= j) for j in range(1, n)]]
    #     for i in range(n)
    # ]
    # columns = list(zip(*non_zero_tuples))
    # tensor = torch.sparse_coo_tensor(torch.tensor(columns), [1] * n).log()
    # return TensorFactor(ordinal, *binaries, tensor=tensor)
    # raise NotImplementedError("we don't want to let you")
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
