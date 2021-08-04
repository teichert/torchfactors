import pytest
import torchfactors as tx
from torchfactors import Domain, Range, SeqDomain, domain
from torchfactors.model import Model
from torchfactors.testing import DummySubject


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


def test_flex_domain():
    d = tx.FlexDomain('domain', unk=True)
    assert d.get_id('test') == 1
    assert d.get_id('test2') == 2
    assert d.get_id('test') == 1
    assert d.get_id('test2') == 2
    d.freeze()
    assert d.get_id('test') == 1
    assert d.get_id('test2') == 2
    assert d.get_id('test3', warn=False) == 0
    assert d.get_id('test') == 1
    assert d.get_id('test2') == 2
    assert d.get_id('test3', warn=False) == 0


def test_domain_to_list():
    model = Model[DummySubject]()
    domain1 = domain.FlexDomain('test1', unk=True)
    values1 = ['this', 'test', 'is', 'a', 'test', 'of', 'this']
    model.domain_ids(domain1, values1)
    out = domain1.to_list()
    assert out == ('test1', True, ['this', 'test', 'is', 'a', 'of'])


def test_domains_from_list():
    input = ('test1', True, ['this', 'test', 'is', 'a', 'of'])
    domain2 = domain.FlexDomain.from_list(input)
    assert domain2.name == 'test1'

    model = Model[DummySubject]()
    values2 = ['test', 'a', 'test', 'of', 'this', 'is', 'this']
    out = model.domain_ids(domain2, values2).tolist()
    assert out == [2, 4, 2, 5, 1, 3, 1]
    assert domain2.frozen


def test_flex_unk():
    d = domain.FlexDomain("Domain")
    with pytest.warns(RuntimeWarning):
        d.get_value(1)
    with pytest.warns(RuntimeWarning):
        d.get_value(-1)
    d.freeze()
    with pytest.warns(RuntimeWarning):
        d.get_id("test")
