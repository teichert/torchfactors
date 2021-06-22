import torch
import torchfactors as tx
from torchfactors import BetheGraph
from torchfactors.factor_graph import FactorGraph
from torchfactors.variable import TensorVar


def test_strategy():
    n = 10
    vs = [
        TensorVar(torch.ones(3, 4), tx.Range(5), tx.LATENT)
        for _ in range(n)
    ]
    factors = [
        tx.TensorFactor([vs[i], vs[i+1]], init=tx.utils.ndrange)
        for i in range(n - 1)
    ]
    fg = FactorGraph(factors)
    bg = BetheGraph(fg)
    assert len(bg.regions) == len(factors) + len(vs)
    assert len(bg.edges) == sum(1 for f in factors for v in f)
