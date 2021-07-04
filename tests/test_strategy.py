import math

import torch
import torchfactors as tx
from torchfactors import BetheGraph
from torchfactors.factor_graph import FactorGraph
from torchfactors.variable import TensorVar


def test_strategy():
    n = 5
    vs = [
        TensorVar(torch.ones(3, 4).float(), tx.Range(5), tx.LATENT)
        for _ in range(n)
    ]
    factors = [
        tx.TensorFactor(vs[i], vs[i+1])
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
    varid = fg.varids[vs[1]]
    expected_regions_with_var1 = [
        (0, bg.regions[0], [vs[1]]),
        (1, bg.regions[1], [vs[1]]),
        (varid, bg.regions[varid], [vs[1]])]

    out_regions_with_var1 = list(bg.get_regions_with_vars(vs[1]))

    assert out_regions_with_var1 == expected_regions_with_var1


def test_strategy_schedule():
    v1 = TensorVar(torch.ones(3, 4).float(), tx.Range(5), tx.LATENT)
    v2 = TensorVar(torch.ones(3, 4).float(), tx.Range(5), tx.LATENT)
    factors = [tx.TensorFactor(v1, v2)]
    fg = FactorGraph(factors)
    bg = BetheGraph(fg)
    schedule = list(bg)
    va, vb = fg.neighbors[0]
    expected_schedule = [
        (0, [va, vb]),
        (0, [va, vb]),
    ]
    assert schedule == expected_schedule
    assert set(bg.reachable_from(0)) == {0, 1, 2}
    assert set(bg.reachable_from(1)) == {1}
    assert set(bg.reachable_from(2)) == {2}

    assert set(bg.penetrating_edges(0)) == set()
    assert set(bg.penetrating_edges(1)) == {(0, 1)}
    assert set(bg.penetrating_edges(2)) == {(0, 2)}


# test_strategy()
