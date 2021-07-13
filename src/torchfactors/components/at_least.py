from typing import Sequence

import torch
from torchfactors.variable import Var

from ..factor import Factor
from .tensor_factor import TensorFactor


def VIsKAndAtLeastB(ordinal: Var, binaries: Sequence[Var]) -> Factor:
    r"""
    only has non-zero probability on configuration between a
    n-ary and (n-1) binary variables for each value k when
    at least the
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
    n = len(ordinal.domain)
    non_zero_tuples = [
        [i, *[int(i >= j) for j in range(1, n)]]
        for i in range(n)
    ]
    columns = list(zip(*non_zero_tuples))
    tensor = torch.sparse_coo_tensor(torch.tensor(columns), [1] * n).log()
    return TensorFactor(ordinal, *binaries, tensor=tensor)
    # raise NotImplementedError("we don't want to let you")
