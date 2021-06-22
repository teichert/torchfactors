# import torch
# from torchfactors import Range, TensorFactor, TensorVar


# def test_factor():
#     t = torch.ones(3, 4)
#     v = TensorVar(t, domain=Range(10))
#     f = TensorFactor(v)
#     assert f.shape == (3, 4, 10)
#     assert f.batch_cells == 3 * 4
#     assert f.batch_shape == (3, 4)
#     assert f.num_batch_dims == 2
#     assert f.out_shape == (10,)
#     assert f.cells == 3 * 4 * 10
#     assert f.free_energy

#     f.tensor = torch.arange(3, 4, 10)
#     f.normalize()
#     assert f.tensor.exp().sum() == 1.0
