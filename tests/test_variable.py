from torchfactors import VarUsage


def test_usage1():
    assert VarUsage.DEFAULT == VarUsage.OBSERVED
