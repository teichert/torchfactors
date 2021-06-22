# import torch
# from torchfactors import Range, TensorFactor, Var


# def test_factor():
#     t = torch.ones(3, 4)
#     v = Var(t, domain=Range(5))
#     f = TensorFactor(v)
#     assert f.shape == (3, 4, 5)
