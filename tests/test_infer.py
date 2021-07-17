import math

import pytest
import torch
from torch import tensor
from torchfactors import ANNOTATED, BP, Range, TensorVar
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


def test_predict_simple():
    coin = TensorVar(Range(2), torch.tensor(0), ANNOTATED)
    factor = TensorFactor(coin, tensor=torch.tensor([1, 3]).log())
    factors = [factor]
    bp = BP()
    assert coin.tensor == 0
    bp.predict(factors)
    assert coin.tensor == 1
    factor.tensor = torch.tensor([3, 1]).log()
    bp.predict(factors)
    assert coin.tensor == 0


test_predict_simple()


def test_predict_simple2():
    coin = TensorVar(Range(2), torch.tensor(0), ANNOTATED)
    factor = TensorFactor(coin, tensor=torch.tensor([0, 1]).log())
    factors = [factor]
    bp = BP()
    assert coin.tensor == 0
    # TODO: waht do I do about this? predict wants to change the variable,
    # but there are factors that are depending on them???
    bp.predict(factors)
    assert coin.tensor == 1
    factor.tensor = torch.tensor([1, 0]).log()
    bp.predict(factors)
    assert coin.tensor == 0


def test_predict_simple3():
    a = TensorVar(Range(2), torch.tensor(0), ANNOTATED)
    b = TensorVar(Range(2), torch.tensor(0), ANNOTATED)
    factor = TensorFactor(a, tensor=torch.tensor([0, 1]).log())
    factors = [
        factor,
        TensorFactor(a, b, tensor=torch.eye(2).log())
    ]
    bp = BP()
    bp.predict(factors)
    assert a.tensor == 1
    assert b.tensor == 1

    factor.tensor = torch.tensor([1, 0]).log()
    bp.predict(factors)
    assert a.tensor == 0
    assert b.tensor == 0


def test_predict_multi():
    bits = TensorVar(Range(2), torch.tensor([False, True, False]), ANNOTATED)
    factors = [
        TensorFactor(bits[..., 0], tensor=torch.tensor([0, 0.5]).log()),
        TensorFactor(bits[..., 1], tensor=torch.tensor([0.5, 0]).log()),
        TensorFactor(bits[..., 2], tensor=torch.tensor([0.5, 0]).log()),
    ]
    bp = BP()
    marginals = bp.product_marginal(factors, bits)
    assert marginals.allclose(torch.tensor([
        [0, 1],
        [1, 0],
        [1, 0]
    ]).log())
    bp.predict(factors)
    assert (bits.tensor == torch.tensor([True, False, False])).all()


# # TODO: make this work;
# # it would be nice to be able to specify models this way; it would be great to
# # be able to execute inference more efficiently by passing messages in parallel
# def test_predict_multi2():
#     bits = TensorVar(Range(2), torch.tensor([False, True, True]), ANNOTATED)
#     factors = [
#         TensorFactor(bits, tensor=torch.tensor([0, 1]).log()),
#         TensorFactor(bits[..., :-1], bits[..., 1:],
#                      tensor=torch.tensor([[0, 1], [1, 0.]]).log()),
#     ]
#     bp = BP(passes=6)
#     marginals = bp.product_marginal(factors, bits)
#     assert marginals.allclose(torch.tensor([
#         [0, 1],
#         [1, 0],
#         [0, 1]
#     ]).log())
#     bp.predict(factors)
#     assert (bits.tensor == torch.tensor([True, False, True])).all()


# test_predict_multi2()


def test_bp_change():
    bits = TensorVar(Range(2), torch.tensor([False]), ANNOTATED)
    factors = [
        TensorFactor(bits[..., 0], tensor=torch.tensor([0, 0.5]).log()),
    ]
    bp = BP()
    _, change = bp.partition_with_change(factors)
    assert change.sum() == 0.0


def test_bp_change2():
    bits = TensorVar(Range(2), torch.tensor([False]), ANNOTATED)
    factors = [
        TensorFactor(bits[..., 0], tensor=torch.tensor([0.25, 0.75]).log()),
    ]
    bp = BP(passes=1)
    _, change = bp.partition_with_change(factors)
    assert change.sum() > 0.0
