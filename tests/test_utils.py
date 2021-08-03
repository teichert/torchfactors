import pandas as pd
import pytest
import torch
import torchfactors as tx
from torch import arange
from torchfactors.subject import ListDataset
from torchfactors.types import gdrop
from torchfactors.utils import (as_ndrange, canonical_ndslice,
                                canonical_range_slice, compose, compose_single,
                                ndarange, ndslices_cat, ndslices_overlap,
                                outer, stereotype)


def test_compose_single():
    out = compose_single(slice(3, 14, 2), slice(2, 6, 3), 100)
    expected = slice(7, 14, 6)
    assert out == expected


def test_compose():
    shape = (4, 10, 12, 7)
    first = (slice(None), 3, slice(3, 9))
    second = (2, slice(1, 3), slice(5))
    expected_combined = (2, 3, slice(4, 6, 1), slice(0, 5, 1))
    out = compose(first, second, shape)
    assert out == expected_combined

    other_first = (slice(2, 4), slice(None), slice(2, 7))
    other_second = (0, 3, slice(2, -1), slice(5))
    other_out = compose(other_first, other_second, shape)
    assert other_out == expected_combined

# def test_compose2():
#     shape = (4, 5)
#     first = (slice(1, 4), slice(None))
#     second = (tx.gslice(2, 0, 1), 1)
#     expected_combined = (tx.gslice(3, 1, 2), 1)

#     assert compose(shape, first, second) == expected_combined


# def test_compose3a():
#     shape = (6,)
#     first = ([2, 1, 1],)
#     second = ([0, 2],)
#     expected_combined = (slice(2, 0, -1),)

#     out = compose(shape, first, second)
#     assert out == expected_combined


# def test_compose3():
#     shape = (4, 5, 6)
#     first = (slice(1, 4), slice(None), [2, 1, 1])
#     second = ([2, 0, 1], 1, [0, 2])
#     expected_combined = ((3, 1, 2), 1, slice(2, 0, -1))

#     out = compose(shape, first, second)
#     assert out == expected_combined


def test_ndrange():
    t = ndarange(2, 3, 4)
    assert (t == arange(2*3*4).reshape(2, 3, 4)).all()


def test_ndslices_overlap():
    shape = (10, 23, 8, 10, 15)
    t = ((3, slice(None), slice(3, 5)))
    assert ndslices_overlap(t, (3, 5, slice(4, 8)), shape)
    assert not ndslices_overlap(t, (4, 5, slice(4, 8)), shape)
    assert ndslices_overlap(t, (slice(None), slice(None), slice(4, 8)), shape)
    assert not ndslices_overlap(t, (slice(None), slice(None), 5), shape)
    assert not ndslices_overlap(t, (slice(None), slice(
        None), slice(None), slice(None), slice(3, 3)), shape)


# def test_ndslice_bad():
#     with pytest.raises(NotImplementedError):
#         as_ndrange(8, shape=(10,))


def test_asndrange_dots():
    shape = (10, 11, 12, 13, 14)
    out = as_ndrange((slice(3, 6), ..., 5), shape=shape)
    assert out == (range(3, 6), range(11), range(12), range(13), 5)


# def test_asndrange_lists():
#     shape = (10, 11, 12, 13, 14)
#     out = as_ndrange((slice(3, 6), [3, 1, 2], slice(None), (3, 4), 5), shape=shape)
#     assert out == (range(3, 6), (3, 1, 2), range(12), range(3, 5), 5)


def test_asndrange_bad_dots():
    shape = (10, 11, 12, 13, 14)
    with pytest.raises(ValueError):
        as_ndrange((slice(3, 6), ..., 2, ..., 5), shape=shape)


def test_ndslices_cat():
    assert ndslices_cat(1, 3) == (1, 3)
    assert ndslices_cat(1, (slice(None), 3)) == (1, slice(None), 3)
    assert ndslices_cat((1, 3), 2) == (1, 3, 2)
    assert ndslices_cat((1, 3), (9, 2)) == (1, 3, 9, 2)


def test_outer2():
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([2, 3])
    out = outer([a, b])
    assert out.allclose(torch.tensor([
        [2, 3],
        [4, 6],
        [6, 9]
    ]))


def test_outer3():
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([2, 3])
    c = torch.tensor([7, 9])
    out = outer([a, b, c])
    assert out.allclose(torch.tensor([
        [[14, 18], [21, 27]],
        [[28, 36], [42, 54]],
        [[42, 54], [63, 81]]
    ]))


def test_outer_batch():
    a = torch.tensor([[1, 2, 3], [1, 2, 6]])
    b = torch.tensor([[2, 3], [2, 3]])
    c = torch.tensor([[7, 9], [7, 9]])
    out = outer([a, b, c], num_batch_dims=1)
    expected = torch.tensor([[
        [[14, 18], [21, 27]],
        [[28, 36], [42, 54]],
        [[42, 54], [63, 81]]
    ], [
        [[14, 18], [21, 27]],
        [[28, 36], [42, 54]],
        [[84, 108], [126, 162]]
    ]])
    assert out.allclose(expected)


# def test_other_outer3():
#     a = torch.tensor([1, 2, 3])
#     b = torch.tensor([2, 3])
#     c = torch.tensor([7, 9])
#     out = tx.utils.outer2([a, b, c])
#     assert out.allclose(torch.tensor([
#         [[14, 18], [21, 27]],
#         [[28, 36], [42, 54]],
#         [[42, 54], [63, 81]]
#     ]))


# def test_other_outer_batch():
#     a = torch.tensor([[1, 2, 3], [1, 2, 6]])
#     b = torch.tensor([[2, 3], [2, 3]])
#     c = torch.tensor([[7, 9], [7, 9]])
#     out = outer(a, b, c, num_batch_dims=1)
#     expected = torch.tensor([[
#         [[14, 18], [21, 27]],
#         [[28, 36], [42, 54]],
#         [[42, 54], [63, 81]]
#     ], [
#         [[14, 18], [21, 27]],
#         [[28, 36], [42, 54]],
#         [[84, 108], [126, 162]]
#     ]])
#     assert out.allclose(expected)


def test_stereotype():
    scales = torch.tensor([2, 3, 4])
    binary = torch.tensor([5, 6])
    out = stereotype(scales, binary)
    # total = scales.sum() * binary[0]
    # full = scales * binary[1] + total
    # for each coordinate, iterate over the 4
    expected = torch.tensor([2*6+9*5, 3*6+9*5, 4*6+9*5])
    assert out.allclose(expected)


# def test_canonical_maybe_range():
#     out = canonical_maybe_range((4, 4))
#     assert out == (4, 4)

def test_canonical_maybe_range2():
    out = canonical_range_slice(gdrop(4, 4))
    assert out == 4


def test_stereotype2():
    scales = torch.tensor([
        [1, 2, 3],
        [3, 4, 5],
        [6, 7, 8],
        [9, 10, 11]
    ]).float()
    binary = torch.tensor([
        [12, 13],
        [14, 15]
    ])
    out = stereotype(scales, binary)
    total = scales.sum() * binary[0, 0]
    full = scales * binary[1, 1] + total
    cols = scales.sum(0) * binary[0, 1]
    rows = scales.sum(1) * binary[1, 0]
    # for each coordinate, iterate over the 4
    expected = torch.tensor(
        [
            [
                full[0, 0] + rows[0] + cols[0],
                full[0, 1] + rows[0] + cols[1],
                full[0, 2] + rows[0] + cols[2],
            ],
            [
                full[1, 0] + rows[1] + cols[0],
                full[1, 1] + rows[1] + cols[1],
                full[1, 2] + rows[1] + cols[2],
            ],
            [
                full[2, 0] + rows[2] + cols[0],
                full[2, 1] + rows[2] + cols[1],
                full[2, 2] + rows[2] + cols[2],
            ],
            [
                full[3, 0] + rows[3] + cols[0],
                full[3, 1] + rows[3] + cols[1],
                full[3, 2] + rows[3] + cols[2],
            ],
        ])
    assert out.allclose(expected)


def test_logsumexp_default():
    t = torch.tensor([[1, 2], [3, 4]]).log()
    out = tx.logsumexp(t)
    assert out == torch.logsumexp(t, (0, 1))


def test_logsumexp_none():
    t = torch.tensor([[1, 2], [3, 4]]).log()
    out = tx.logsumexp(t, ())
    assert out.allclose(t)


def test_logsumexp_none_out():
    t = torch.tensor([[1, 2], [3, 4]]).log()
    out = torch.zeros_like(t)
    tx.logsumexp(t, (), out=out)
    assert out.allclose(t)


def test_logsumexp_two():
    t = torch.tensor([[1, 2], [3, 4]]).log()
    out = tx.logsumexp(t, (0, 1))
    assert out == t.logsumexp((0, 1))


def test_logsumexp_one():
    t = torch.tensor([[1, 2], [3, 4]]).log()
    out = tx.logsumexp(t, 1)
    assert out.allclose(t.logsumexp(1))


def test_logsumexp_one_tuple():
    t = torch.tensor([[1, 2], [3, 4]]).log()
    out = tx.logsumexp(t, (1,))
    assert out.allclose(t.logsumexp(1))


def test_gdrop():
    a = tx.gdrop(3, 2, 5)
    assert a.indexPerIndex == (3, 2, 5)


def test_gdrop2():
    a = tx.gdrop([3, 2, 5])
    assert a.indexPerIndex == (3, 2, 5)


def test_gdrop3():
    a = tx.gdrop((3, 2, 5))
    assert a.indexPerIndex == (3, 2, 5)


def test_end():
    assert tx.utils.end(5, 3) == 6
    assert tx.utils.end(5, -3) == 4
    assert tx.utils.end(-5, 3) == -4
    assert tx.utils.end(-5, -3) == -6


def test_canonical_range_slice():
    assert canonical_range_slice(5) == 5
    assert canonical_range_slice(range(5)) == range(5)
    assert canonical_range_slice(range(3, 20, 10)) == range(3, 14, 10)
    assert canonical_range_slice(range(20, 3, -10)) == range(20, 9, -10)
    assert canonical_range_slice(tx.gdrop(3, 7, 2)) == tx.gdrop(3, 7, 2)
    assert canonical_range_slice(tx.gdrop(3, 3, 3)) == 3
    assert canonical_range_slice(range(10, 10)) == range(0)
    with pytest.raises(TypeError):
        # don't know how to canonicalize a string
        canonical_range_slice("test")  # type: ignore


def test_canonical_ndslice():
    assert canonical_ndslice(..., (8,)) == (...,)  # type: ignore
    assert canonical_ndslice(4, (8,)) == (4,)
    out = canonical_ndslice(slice(4, 9), (12,))
    expected = (slice(4, 9, 1),)
    assert out == expected


def test_canonical_ndslice2():
    ndslice = (slice(4, 9), 3, slice(-1, 3, -10), 4)
    shape = (12, 5, 30, 8)
    out = canonical_ndslice(ndslice, shape)
    expected = (slice(4, 9, 1), 3, slice(29, 8, -10), 4)
    assert out == expected


def test_bad_compose():
    with pytest.raises(ValueError):
        compose_single(4, slice(4, 2), 10)


def test_gdrop_single():
    out = compose_single(slice(5, 8), tx.gdrop(4, 2, 7), 10)
    assert out == tx.gdrop(4, 2, 7)


def test_gdrop_multi():
    shape = (10, 15, 11, 12, 13)
    a = (5, slice(5, 8), slice(None, 8), slice(None), 3)  # 3 x 11 x 12
    b = (slice(None), tx.gdrop(4, 2, 7), slice(None))  # 3 x 12
    expected = (5, slice(5, 8, 1), tx.gdrop(4, 2, 7), slice(0, 12, 1), 3)
    out = compose(a, b, shape)
    assert out == expected


def test_data_len():
    data = ListDataset([tx.testing.DummySubject()] * 5)
    assert tx.data_len(data) == 5


def test_with_rep_number():
    orig = dict(
        sentence=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        property=[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
        annotator=[1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1],
        label=[1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2]
    )
    df = pd.DataFrame(orig)
    with_rep = tx.extra_utils.with_rep_number(
        df, group_key=['sentence', 'property'], name='my_rep')
    expected = dict(
        sentence=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        property=[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
        annotator=[1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1],
        label=[1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2],
        my_rep=[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    out = with_rep.to_dict(orient='list')
    assert out == expected
    assert df.to_dict(orient='list') == orig
