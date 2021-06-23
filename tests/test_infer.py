import torch
from torch import tensor
from torchfactors import (BetheGraph, FactorGraph, Range, TensorVar,
                          product_marginal)
from torchfactors.components.tensor_factor import TensorFactor


def test_infer():
    a = TensorVar(tensor(2), Range(10))
    b = TensorVar(tensor(3), Range(9))
    c = TensorVar(tensor(4), Range(8))
    fg = FactorGraph([
        TensorFactor(a),
        TensorFactor(b),
        TensorFactor(c),
        TensorFactor([a, b]),
        TensorFactor([b, c])])
    strategy = BetheGraph(fg)
    logz = product_marginal(fg, strategy=strategy)
    assert logz.exp().isclose(torch.tensor(10. * 9 * 8))


test_infer()
