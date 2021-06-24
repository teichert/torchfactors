import math

import pytest
import torch
from torch import tensor
from torchfactors import BP, Range, TensorVar
from torchfactors.components.tensor_factor import TensorFactor


def test_infer():
    a = TensorVar(tensor(2), Range(10))
    b = TensorVar(tensor(3), Range(9))
    c = TensorVar(tensor(4), Range(8))
    factors = [
        TensorFactor(a),
        TensorFactor(b),
        TensorFactor(c),
        TensorFactor(a, b),
        TensorFactor(b, c)]
    bp = BP()
    logz = bp.product_marginal(factors)
    assert logz.exp().isclose(torch.tensor(10. * 9 * 8))

    marg_a = bp.product_marginal(factors, a)
    assert (marg_a == -math.log(10)).all()

    marg_a = bp.product_marginal(factors, a)
    assert (marg_a == -math.log(10)).all()

    with pytest.raises(ValueError):
        # should have used product_marginal instead
        bp.product_marginals(factors, a)  # type: ignore

    with pytest.raises(ValueError):
        bp.product_marginal(factors, (a, b))

    marg_a = bp.product_marginal(factors, a, normalize=False)
    assert (marg_a == math.log(9*8)).all()

    logz, marg_a = bp.product_marginals(factors, (), (a,), normalize=True)
    assert logz.exp().isclose(torch.tensor(10. * 9 * 8))
    assert (marg_a == -math.log(10)).all()

    logz, marg_a = bp.product_marginals(factors, (), (a,), normalize=False)
    assert logz.exp().isclose(torch.tensor(10. * 9 * 8))
    assert (marg_a == math.log(9*8)).all()
