

import pytest
import torch

import torchfactors as tx
from torchfactors import inferencer
from torchfactors.clique import (BinaryScoresModule,
                                 make_binary_label_variables,
                                 make_binary_threshold_variables)
from torchfactors.components.at_least import KIsAtLeastJ
from torchfactors.components.binary import Binary
from torchfactors.components.nominal import Nominal
from torchfactors.components.prop_odds import ProportionalOdds
from torchfactors.components.stereotype import Stereotype
from torchfactors.components.tensor_factor import \
    linear_binary_to_ordinal_tensor
from torchfactors.subject import Environment
from torchfactors.testing import DummyParamNamespace, DummyVar
from torchfactors.utils import num_trainable
from torchfactors.variable import VarField


def test_make_binary_label_variables():
    a = tx.TensorVar(torch.tensor([0, 1, 2, 4]), tx.OBSERVED, tx.Range(5))
    b = DummyVar(4, domain_size=6)
    c = DummyVar(5, domain_size=2)
    env = Environment()
    vars = make_binary_label_variables(env, a, b, c, key='root')

    # shouldn't model the 0 value
    with pytest.raises(KeyError):
        vars[a, 0]
    assert vars[a, 1].tensor.tolist() == [0, 1, 1, 1]
    assert vars[a, 2].tensor.tolist() == [0, 0, 1, 1]
    assert vars[a, 3].tensor.tolist() == [0, 0, 0, 1]
    assert vars[a, 4].tensor.tolist() == [0, 0, 0, 1]
    assert len(vars) == (5 - 1) + (6 - 1) + (2 - 1)
    assert len(vars[a, 2].domain) == 2
    assert len(vars[c, 1].domain) == 2
    assert env.variable((a, 2, 'root')) == vars[a, 2]


def test_make_latent_binary_label_variables():
    a = DummyVar(3)
    env = Environment()
    vars = make_binary_label_variables(env, a, key='root', latent=True)
    assert (vars[a, 1].usage == tx.LATENT).all()


def test_make_equals_binary_label_variables():
    a = tx.TensorVar(torch.tensor([0, 1, 2, 4]), tx.OBSERVED, tx.Range(5))
    b = DummyVar(4, domain_size=6)
    c = DummyVar(5, domain_size=2)
    env = Environment()
    vars = make_binary_label_variables(env, a, b, c, key='root', only_equals=True)
    assert vars[a, 1].tensor.tolist() == [0, 1, 0, 0]
    assert vars[a, 2].tensor.tolist() == [0, 0, 1, 0]
    assert vars[a, 3].tensor.tolist() == [0, 0, 0, 0]
    assert vars[a, 4].tensor.tolist() == [0, 0, 0, 1]


def test_make_threshold_variables():
    a = tx.TensorVar(torch.tensor([3, 0, 2, 4, 1]), tx.OBSERVED, tx.Range(5))
    b = tx.TensorVar(torch.tensor([3, 0, 5, 2, 4, 1]), tx.OBSERVED, tx.Range(6))
    c = DummyVar(5, domain_size=2)
    env = Environment()
    vars = make_binary_threshold_variables(env, a, b, c, key='root')
    assert len(vars) == 3
    # posiitive a should be top two lables: 3 and 4
    assert vars[a].tensor.tolist() == [1, 0, 0, 1, 0]
    # posiitive b should be top three lables: 3, 4, and 5
    assert vars[b].tensor.tolist() == [1, 0, 1, 0, 1, 0]


def test_make_latent_threshold_variables():
    a = tx.TensorVar(torch.tensor([3, 0, 2, 4, 1]), tx.OBSERVED, tx.Range(5))
    b = tx.TensorVar(torch.tensor([3, 0, 5, 2, 4, 1]), tx.LATENT, tx.Range(6))
    c = tx.TensorVar(torch.tensor([3, 0, 5, 2, 4, 1]), tx.PADDING, tx.Range(6))
    env = Environment()
    vars_obs = make_binary_threshold_variables(env, a, b, c, key='root')
    assert (vars_obs[a].usage == tx.OBSERVED).all()
    assert (vars_obs[b].usage == tx.LATENT).all()
    assert (vars_obs[c].usage == tx.PADDING).all()
    env = Environment()
    vars_lat = make_binary_threshold_variables(env, a, b, c, key='root', latent=True)
    assert (vars_lat[a].usage == tx.LATENT).all()
    assert (vars_lat[b].usage == tx.LATENT).all()
    assert (vars_lat[c].usage == tx.PADDING).all()


def test_binary_scores_module():
    params = DummyParamNamespace()
    a = tx.TensorVar(torch.tensor([3, 0, 2, 4, 1]), tx.OBSERVED, tx.Range(5))
    b = tx.TensorVar(torch.tensor([3, 0, 5, 2, 4, 1]), tx.LATENT, tx.Range(6))
    c = tx.TensorVar(torch.tensor([3, 0, 5, 2, 4, 1]), tx.PADDING, tx.Range(6))
    input_a = torch.tensor([1, 2, 3]).float()
    module = BinaryScoresModule(params, [a, b, c], (3,))
    out: torch.Tensor = module(input_a)
    module2 = BinaryScoresModule(params, [a, b, c], (3,))
    out2: torch.Tensor = module2(input_a)
    assert out.allclose(out2)
    input_b = torch.tensor([3, 2, 1]).float()
    out3: torch.Tensor = module2(input_b)
    assert not out.allclose(out3)


def testKIsAtLeastJ():
    a = tx.TensorVar(torch.tensor([3, 0, 2, 1, 3]), tx.ANNOTATED, tx.Range(4))
    a1 = tx.TensorVar(torch.zeros(5), tx.ANNOTATED, tx.Range(2))
    a3 = tx.TensorVar(torch.zeros(5), tx.ANNOTATED, tx.Range(2))
    factor1 = KIsAtLeastJ(a, a1, 1)
    out1 = factor1.dense
    assert out1.shape == (5, 4, 2)
    expected1 = torch.tensor(  # j = 1
        [
            [  # batch element = 0
                [1, 0],  # k = 0, (k not at least j, is at least)
                [0, 1],  # k = 1, (1 is not at least 1?, 1 is at least 1?)
                [0, 1],  # k = 2,
                [0, 1],  # k = 3
            ],
        ] * 5).float()
    assert out1.exp().allclose(expected1)

    expected3 = torch.tensor(  # j = 3
        [
            [  # batch element = 0
                [1, 0],  # k = 0, (k not at least j, is at least)
                [1, 0],  # k = 1, (1 is not at least 3?, 1 is at least 3?)
                [1, 0],  # k = 2,
                [0, 1],  # k = 3
            ],
        ] * 5).float()
    factor3 = KIsAtLeastJ(a, a3, 3)
    out3 = factor3.dense
    assert out3.shape == (5, 4, 2)
    assert out3.exp().allclose(expected3)


def test_binary():
    a = tx.TensorVar(torch.tensor([3, 0, 2, 1, 3]), tx.ANNOTATED, tx.Range(4), ndims=0)
    assert tuple(a.marginal_shape) == (5, 4)
    b = tx.TensorVar(torch.tensor([1, 1, 2, 2, 0]), tx.ANNOTATED, tx.Range(3), ndims=0)
    assert tuple(b.marginal_shape) == (5, 3)

    env = Environment()
    params = DummyParamNamespace()
    model = Binary(latent=False)
    input = torch.ones(5, 7)
    factors = list(model.factors(env, params, a, b, input=input))
    assert len(factors) == 1
    # single binary factor
    assert factors[0].shape == (5, 4, 3)
    # same for all inputs (which are the same)
    d = factors[0].dense
    assert d.allclose(d.mean(dim=0))

    # assert factors[1].shape == (5, 4, 2)
    # assert factors[2].shape == (5, 3, 2)

    # assert len(factors[0]) == 2
    # bin_a, bin_b = factors[0].variables
    # assert len(bin_a.domain) == 2
    # # assert tuple(bin_a.marginal_shape) == (5, 2)
    # assert tuple(bin_b.marginal_shape) == (5, 2)
    # vars1 = factors[1].variables
    # assert tuple(vars1) == (a, bin_a)
    # vars2 = factors[2].variables
    # assert tuple(vars2) == (b, bin_b)

    [f.dense for f in factors]
    out_params = num_trainable(params.model)
    # binary configs (4) * (features plus bias) + binary to label mapping (for each)
    # expected_params = 4 * (7 + 1) + 2 * (4 + 3)
    # expected_params = 4 * (7 + 1) + 2 * (4 + 3)

    # just a single param saying how much for both to be on
    assert out_params == 1


def test_linear_binary_to_ordinal_2():
    assert linear_binary_to_ordinal_tensor(2).allclose(torch.tensor([
        [0, 0],
        [0, 1.0]
    ]))


def test_linear_binary_to_ordinal_3():
    assert linear_binary_to_ordinal_tensor(3).allclose(torch.tensor([
        [0, 0, 0],
        [0, 0.5, 1.0]
    ]))


def test_linear_binary_to_ordinal_4():
    assert linear_binary_to_ordinal_tensor(4).allclose(torch.tensor([
        [0, 0, 0, 0],
        [0, 1/3, 2/3, 1.0]
    ]))


def test_linear_binary_to_ordinal_5():
    assert linear_binary_to_ordinal_tensor(5).allclose(torch.tensor([
        [0, 0, 0, 0, 0],
        [0, 0.25, 0.5, 0.75, 1.0]
    ]))


def test_latent_binary():
    env = Environment()
    model = Binary(latent=True)
    params = DummyParamNamespace()
    input = torch.ones(5, 8)
    a = tx.TensorVar(torch.tensor([3, 0, 2, 1, 3]), tx.ANNOTATED, tx.Range(4), ndims=0)
    b = tx.TensorVar(torch.tensor([1, 1, 2, 2, 0]), tx.ANNOTATED, tx.Range(3), ndims=0)

    def factors():
        yield from model.factors(env, params.namespace('a,b'), a, b, input=input)
        yield from model.factors(env, params.namespace('a'), a, input=input)
        yield from model.factors(env, params.namespace('b'), b, input=input)
    fs = list(factors())
    ts = [f.dense for f in fs if f is not None]
    # one pairwise and two unary factors on the latent binary, plus one binary
    # to ordinal for each variable,
    assert len(ts) == 1 + 2 + 2
    expected_params = (
        8 +  # minimal weight between a and b latent binaries
        8 +  # minimal weight on a latent binary
        8 +  # minimal weight on b latent binary
        3 +  # minimal bias on those three factors
        3 +  # minimal bias between latent binary and a
        2)  # minimal bias between latent binary and b
    out_params = num_trainable(params.model)
    assert out_params == expected_params


def test_latent_binary_linear():
    env = Environment()
    model = Binary(latent=True, linear=True)
    params = DummyParamNamespace()
    input = torch.ones(5, 8)
    a = tx.TensorVar(torch.tensor([3, 0, 2, 1, 3]), tx.ANNOTATED, tx.Range(4), ndims=0)
    b = tx.TensorVar(torch.tensor([1, 1, 2, 2, 0]), tx.ANNOTATED, tx.Range(3), ndims=0)

    def factors():
        yield from model.factors(env, params.namespace('a,b'), a, b, input=input)
        yield from model.factors(env, params.namespace('a'), a, input=input)
        yield from model.factors(env, params.namespace('b'), b, input=input)
    fs = list(factors())
    ts = [f.dense for f in fs if f is not None]
    # one pairwise and two unary factors on the latent binary, plus one binary
    # to ordinal for each variable,
    assert len(ts) == 1 + 2 + 2
    expected_params = (
        8 +  # minimal weight between a and b latent binaries
        8 +  # minimal weight on a latent binary
        8 +  # minimal weight on b latent binary
        3)  # minimal bias on those three factors
    out_params = num_trainable(params.model)
    assert out_params == expected_params


def test_nominal():
    env = Environment()
    model = Nominal()
    params = DummyParamNamespace()
    input = torch.ones(5, 8)
    a = tx.TensorVar(torch.tensor([3, 0, 2, 1, 3]), tx.ANNOTATED, tx.Range(4), ndims=0)
    b = tx.TensorVar(torch.tensor([1, 1, 2, 2, 0]), tx.ANNOTATED, tx.Range(3), ndims=0)
    factors = list(model.factors(env, params, a, b, input=input))
    [f.dense for f in factors]

    out_params = num_trainable(params.model)
    # configs times (labels plus bias)
    expected_params = 4 * 3 * (8 + 1)
    assert out_params == expected_params

    assert len(factors) == 1
    f, = factors
    assert len(f) == 2
    v1, v2 = f.variables
    assert v1 is a
    assert v2 is b

def test_prop_odds_single():
    env = Environment()
    model = ProportionalOdds()
    params = DummyParamNamespace()
    input = torch.ones(5, 9)
    a = tx.TensorVar(torch.tensor([3, 0, 2, 1, 3]), tx.ANNOTATED, tx.Range(4), ndims=0)
    factors = list(model.factors(env, params, a, input=input))
    densed = [f.dense for f in factors]
    # one bias for for the entire thing; one binary and mapping for all but one value
    assert len(factors) == 1 + 3 * 2
    assert len(factors) == len(densed)
    assert all(f.shape == (5, 4) for f in factors[:1])
    assert all(f.shape == (5, 2) for f in factors[1::2])
    assert all(f.shape == (5, 4, 2) for f in factors[2::2])

    out_params = num_trainable(params.model)
    # features * num_bin_configs + num_full_configs for bias
    expected_params = 9 + 9
    assert out_params == expected_params

    inferencer = tx.BruteForce()
    # sys = tx.System(model, )
    loglikelihood, _ = inferencer.partition_with_change(factors)
    print(loglikelihood)
    loglikelihood.backward()
    print(list(params.model.parameters()))
    print('done')


def test_prop_odds():
    env = Environment()
    model = ProportionalOdds()
    params = DummyParamNamespace()
    input = torch.ones(5, 9)
    a = tx.TensorVar(torch.tensor([3, 0, 2, 1, 3]), tx.ANNOTATED, tx.Range(4), ndims=0)
    b = tx.TensorVar(torch.tensor([1, 1, 2, 2, 0]), tx.ANNOTATED, tx.Range(3), ndims=0)
    factors = list(model.factors(env, params, a, b, input=input))
    [f.dense for f in factors]
    # pairing 3 binary to other 2 binary and adding in mapping for each
    assert len(factors) == 3 * 2 + 3 + 2
    assert all(f.shape == (5, 2, 2) for f in factors[:-5])
    assert all(f.shape == (5, 4, 2) for f in factors[6:9])
    assert all(f.shape == (5, 3, 2) for f in factors[9:])

    out_params = num_trainable(params.model)
    # features * num_bin_configs + num_full_configs for bias
    expected_params = 9 * 4 + 4 * 3
    assert out_params == expected_params

    inferencer = tx.BruteForce()
    # sys = tx.System(model, )
    loglikelihood, _ = inferencer.partition_with_change(factors)
    print(loglikelihood)
    loglikelihood.backward()
    print(list(params.model.parameters()))
    print('done')


def test_non_linear_stereotype():
    env = Environment()
    model = Stereotype(linear=False)
    params = DummyParamNamespace()
    input = torch.ones(5, 9)
    a = tx.TensorVar(torch.tensor([3, 0, 2, 1, 3]), tx.ANNOTATED, tx.Range(4), ndims=0)
    b = tx.TensorVar(torch.tensor([1, 1, 2, 2, 0]), tx.ANNOTATED, tx.Range(3), ndims=0)
    factors = [f.dense for f in model.factors(env, params, a, b, input=input)]
    assert len(factors) == 2

    out_params = num_trainable(params.model)
    # features * num_bin_configs + num_full_configs for bias + scale for each config
    expected_params = 9 * 4 + 4 * 3 + 4 * 3
    assert out_params == expected_params


def test_linear_stereotype():
    env = Environment()
    model = Stereotype()
    params = DummyParamNamespace()
    input = torch.ones(5, 9)
    a = tx.TensorVar(torch.tensor([3, 0, 2, 1, 3]), tx.ANNOTATED, tx.Range(4), ndims=0)
    b = tx.TensorVar(torch.tensor([1, 1, 2, 2, 0]), tx.ANNOTATED, tx.Range(3), ndims=0)
    factors = [f.dense for f in model.factors(env, params, a, b, input=input)]
    assert len(factors) == 2

    out_params = num_trainable(params.model)
    # features * num_bin_configs + num_full_configs for bias
    expected_params = 9 * 4 + 4 * 3
    assert out_params == expected_params


def test_linear_stereotyp_bias_only():
    env = Environment()
    model = Stereotype()
    params = DummyParamNamespace()
    a = tx.TensorVar(torch.tensor([3, 0, 2, 1, 3]), tx.ANNOTATED, tx.Range(4), ndims=0)
    b = tx.TensorVar(torch.tensor([1, 1, 2, 2, 0]), tx.ANNOTATED, tx.Range(3), ndims=0)
    factors = [f.dense for f in model.factors(env, params, a, b, input=None)]
    assert len(factors) == 1

    out_params = num_trainable(params.model)
    # num_full_configs for bias
    expected_params = 4 * 3
    assert out_params == expected_params
