from torch import arange
from torchfactors import ndrange


def test_ndrange():
    t = ndrange(2, 3, 4)
    assert (t == arange(2*3*4).reshape(2, 3, 4)).all()
