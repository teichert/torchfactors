import pytest
from torchfactors import Domain, Range, SeqDomain


def test_seq_domain():
    my_domain: Domain = SeqDomain(['this', 0, 'test'])
    assert len(my_domain) == 3
    assert list(my_domain) == ['this', 0, 'test']


def test_seq_range1():
    my_domain: Domain = Range(3, 10)
    assert len(my_domain) == 7
    assert list(my_domain) == [3, 4, 5, 6, 7, 8, 9]


def test_seq_range2():
    my_domain: Domain = Range(10)
    assert len(my_domain) == 10
    assert list(my_domain) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_seq_range3():
    my_domain: Domain = Range(10)
    assert len(my_domain) == 10
    assert list(my_domain) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_seq_range4():
    my_domain: Domain = Range[:10]
    assert len(my_domain) == 10
    assert list(my_domain) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_seq_range5():
    my_domain: Domain = Range[3:10]
    assert len(my_domain) == 7
    assert list(my_domain) == [3, 4, 5, 6, 7, 8, 9]


def test_domain6():
    assert repr(SeqDomain([3, 4, 1])) == "SeqDomain[3, 4, 1]"


def test_fail_open():
    with pytest.raises(ValueError):
        len(Domain.OPEN)

    with pytest.raises(ValueError):
        iter(Domain.OPEN)
