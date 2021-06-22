import torch
import torchfactors as tx
from torchfactors.factor_graph import FactorGraph
from torchfactors.variable import TensorVar


def test_factor_graph():
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
    assert fg.factors == factors
    assert fg.num_factors == len(factors)
    assert set(fg.variables) == set(vs)
    assert fg.num_edges == (n - 1) * 2
    # variable 1 touches first two factors
    assert fg.neighbors[
        fg.varids[vs[1]]
    ] == [0, 1]
    assert fg.neighbors[1] == [
        fg.varids[vs[1]], fg.varids[vs[2]]
    ]


test_factor_graph()
