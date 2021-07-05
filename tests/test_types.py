import pytest
import torch
from torchfactors.types import ReadOnlyView


def test_readonly():
    t = ReadOnlyView(torch.ones(3, 4))
    assert t.sum() == 12
    with pytest.raises(TypeError):
        t[1] = 2
    with pytest.raises(TypeError):
        t.abs_()
    with pytest.raises(TypeError):
        t.view(t.shape).sum().abs_()
