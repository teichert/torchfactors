import math

import torch
import torchfactors as tx
from torchfactors import BetheGraph
from torchfactors.factor_graph import FactorGraph
from torchfactors.variable import TensorVar


def test_strategy():
    n = 4
    vs = [
        TensorVar(torch.ones(3, 4).float(), tx.Range(5), tx.LATENT)
        for _ in range(n)
    ]
    factors = [
        tx.TensorFactor([vs[i], vs[i+1]])
        for i in range(n - 1)
    ]
    fg = FactorGraph(factors)
    bg = BetheGraph(fg)
    assert len(bg.regions) == len(factors) + len(vs)
    assert len(bg.edges) == sum(1 for f in factors for v in f)
    # corresponds to factor 0 which touches vars 0 and 1
    assert set(bg.regions[0].variables) == {vs[0], vs[1]}
    assert set(bg.regions[0].factors) == {factors[0]}
    assert set(bg.regions[0].factor_set) == {factors[0]}

    out, = bg.regions[0].product_marginals([[vs[0]]])
    assert out.shape == (3, 4, 5)
    assert (out == math.log(5)).all()

    # since the true distribution is uniform, then the free energy without messages should match
    logz = -bg.regions[0].free_energy(())
    assert logz.shape == (3, 4)
    assert logz.exp().isclose(torch.full((3, 4), 25.)).all()
