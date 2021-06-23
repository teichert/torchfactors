import math
from typing import cast

import pytest
import torch
from torch import tensor
from torch.functional import Tensor
from torchfactors import (BetheGraph, FactorGraph, Range, TensorVar,
                          product_marginal, product_marginals)
from torchfactors.components.tensor_factor import TensorFactor


def test_infer():
    a = TensorVar(tensor(2), Range(10))
    b = TensorVar(tensor(3), Range(9))
    c = TensorVar(tensor(4), Range(8))
    fg = FactorGraph([
        TensorFactor(a),
        TensorFactor(b),
        TensorFactor(c),
        TensorFactor(a, b),
        TensorFactor(b, c)])
    strategy = BetheGraph(fg)
    logz = product_marginal(fg, strategy=strategy)
    assert logz.exp().isclose(torch.tensor(10. * 9 * 8))

    marg_a = product_marginal(fg, a, strategy=strategy)
    assert (marg_a == -math.log(10)).all()

    marg_a = product_marginal(fg, a)
    assert (marg_a == -math.log(10)).all()

    with pytest.raises(ValueError):
        # should have used product_marginal instead
        product_marginals(fg, a)

    with pytest.raises(ValueError):
        product_marginal(fg, (a, b), strategy=strategy)

    marg_a = product_marginal(fg, a, normalize=False)
    assert (marg_a == math.log(9*8)).all()

    marg_a = cast(Tensor, product_marginals(fg, (a,), normalize=False, force_multi=False))
    assert (marg_a == math.log(9*8)).all()

    logz, marg_a = product_marginals(fg, (), (a,), normalize=True)
    assert logz.exp().isclose(torch.tensor(10. * 9 * 8))
    assert (marg_a == -math.log(10)).all()

    logz, marg_a = product_marginals(fg, (), (a,), normalize=False)
    assert logz.exp().isclose(torch.tensor(10. * 9 * 8))
    assert (marg_a == math.log(9*8)).all()
